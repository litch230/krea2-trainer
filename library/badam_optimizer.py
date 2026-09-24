import re
import random
import logging
import torch
from torch.optim import Optimizer

# Set up logging for BAdam optimizer diagnostics
logger = logging.getLogger("badam")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[BAdam] %(message)s"))
    logger.addHandler(ch)


class CombinedState(dict):
    """
    Proxy dictionary that merges active optimizer state and offloaded cached optimizer state.
    Provides standard dict access for compatibility with PyTorch optimizer operations.
    """
    def __init__(self, active_state, cached_state):
        super().__init__()
        self.active_state = active_state
        self.cached_state = cached_state

    def __getitem__(self, key):
        if key in self.active_state:
            return self.active_state[key]
        if key in self.cached_state:
            return self.cached_state[key]
        
        # Mimic collections.defaultdict(dict) behavior expected of Optimizer.state.
        # If the parameter requires grad, it's considered active; store the new state dict in active_state.
        # Otherwise, store it in cached_state.
        is_active = getattr(key, "requires_grad", False)
        new_dict = {}
        if is_active:
            self.active_state[key] = new_dict
        else:
            self.cached_state[key] = new_dict
        return new_dict

    def __setitem__(self, key, value):
        if key in self.active_state:
            self.active_state[key] = value
        else:
            self.cached_state[key] = value

    def __delitem__(self, key):
        if key in self.active_state:
            del self.active_state[key]
        if key in self.cached_state:
            del self.cached_state[key]

    def __contains__(self, key):
        return key in self.active_state or key in self.cached_state

    def __len__(self):
        return len(self.keys())

    def get(self, key, default=None):
        if key in self.active_state:
            return self.active_state[key]
        if key in self.cached_state:
            return self.cached_state[key]
        return default

    def keys(self):
        return set(self.active_state.keys()) | set(self.cached_state.keys())

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]

    def __repr__(self):
        return f"CombinedState(active={len(self.active_state)} keys, cached={len(self.cached_state)} keys)"


class BlockPartitioner:
    """
    Provides static partitioning strategies to partition model parameters into blocks.
    """

    @staticmethod
    def partition_by_transformer_layer(named_parameters):
        """
        Groups parameters by parsing transformer layer patterns in their names.
        Supports:
          - Flux double/single blocks
          - Cosmos/Wan blocks
          - DiT layers/joint_blocks
          - UNet input/middle/output blocks
        """
        block_groups = {}
        other_params = []

        # Common block identifier patterns
        patterns = [
            re.compile(r"(.*double_blocks\.\d+)"),
            re.compile(r"(.*single_blocks\.\d+)"),
            re.compile(r"(.*joint_blocks\.\d+)"),
            re.compile(r"(.*layers\.\d+)"),
            re.compile(r"(.*blocks\.\d+)"),
            re.compile(r"(.*input_blocks\.\d+)"),
            re.compile(r"(.*output_blocks\.\d+)"),
            re.compile(r"(.*middle_block)"),
        ]

        for name, param in named_parameters:
            matched = False
            for pattern in patterns:
                m = pattern.match(name)
                if m:
                    block_name = m.group(1)
                    if block_name not in block_groups:
                        block_groups[block_name] = []
                    block_groups[block_name].append(param)
                    matched = True
                    break
            if not matched:
                other_params.append(param)

        def block_key(name):
            numbers = [int(s) for s in re.findall(r"\d+", name)]
            num = numbers[0] if numbers else 0
            
            # Group ordering priority
            if "double_blocks" in name:
                prefix_priority = 0
            elif "single_blocks" in name:
                prefix_priority = 1
            elif "input_blocks" in name:
                prefix_priority = 2
            elif "middle_block" in name:
                prefix_priority = 3
            elif "output_blocks" in name:
                prefix_priority = 4
            elif "joint_blocks" in name:
                prefix_priority = 5
            elif "layers" in name:
                prefix_priority = 6
            elif "blocks" in name:
                prefix_priority = 7
            else:
                prefix_priority = 8
            return (prefix_priority, num, name)

        sorted_block_names = sorted(block_groups.keys(), key=block_key)

        blocks = []
        # Non-transformer parameters form the base block (embeddings, output projections, etc.)
        if other_params:
            blocks.append(other_params)

        for name in sorted_block_names:
            blocks.append(block_groups[name])

        return blocks

    @staticmethod
    def partition_by_regex(named_parameters, regex_patterns):
        """
        Groups parameters by a list of regex patterns. Parameters that match
        the first pattern are grouped in block 1, second in block 2, etc.
        Any remaining unmatched parameters are grouped into the last block.
        """
        if not regex_patterns:
            raise ValueError("regex_patterns must be provided when strategy='regex'")

        compiled = [re.compile(pat) for pat in regex_patterns]
        blocks = [[] for _ in range(len(compiled) + 1)]

        for name, param in named_parameters:
            matched = False
            for idx, pattern in enumerate(compiled):
                if pattern.search(name):
                    blocks[idx].append(param)
                    matched = True
                    break
            if not matched:
                blocks[-1].append(param)

        return [b for b in blocks if b]

    @staticmethod
    def partition_by_param_count(named_parameters, target_param_count):
        """
        Groups parameters sequentially so that each block contains roughly the
        target parameter count (e.g. 100M parameters).
        """
        if not target_param_count or target_param_count <= 0:
            raise ValueError("target_param_count must be positive when strategy='param_count'")

        blocks = []
        current_block = []
        current_count = 0

        for name, param in named_parameters:
            numel = param.numel()
            if current_count + numel > target_param_count and current_block:
                blocks.append(current_block)
                current_block = [param]
                current_count = numel
            else:
                current_block.append(param)
                current_count += numel

        if current_block:
            blocks.append(current_block)

        return blocks

    @staticmethod
    def partition_by_custom(named_parameters, custom_blocks):
        """
        Supports partitioning by custom lists of parameter names, regex patterns, or Parameter objects.
        """
        if not custom_blocks:
            raise ValueError("custom_blocks must be provided when strategy='custom'")

        name_to_param = {name: param for name, param in named_parameters}
        blocks = []
        assigned = set()

        for block_def in custom_blocks:
            block_params = []
            for item in block_def:
                if isinstance(item, torch.Tensor):
                    if item not in assigned:
                        block_params.append(item)
                        assigned.add(item)
                elif isinstance(item, str):
                    if item in name_to_param:
                        p = name_to_param[item]
                        if p not in assigned:
                            block_params.append(p)
                            assigned.add(p)
                    else:
                        # Treat it as a regex pattern
                        try:
                            pat = re.compile(item)
                            for name, param in named_parameters:
                                if pat.search(name) and param not in assigned:
                                    block_params.append(param)
                                    assigned.add(param)
                        except re.error:
                            pass
            if block_params:
                blocks.append(block_params)

        remaining = [param for name, param in named_parameters if param not in assigned]
        if remaining:
            blocks.append(remaining)

        return blocks


class BlockOptimizer(Optimizer):
    """
    BAdam (Block-wise Adam) Optimizer Wrapper.
    Manages parameters by partitioning them into blocks and only training one block at a time.
    """
    def __init__(
        self,
        base_optimizer,
        named_parameters,
        switch_block_every=100,
        switch_mode="ascending",
        block_strategy="transformer_layer",
        offload_to_cpu=True,
        release_inactive_state=False,
        regex_patterns=None,
        target_param_count=None,
        custom_blocks=None,
        active_blocks_count=1,
        **kwargs
    ):
        # We wrap the base optimizer instance.
        self.base_optimizer = base_optimizer
        self.switch_block_every = switch_block_every
        self.switch_mode = switch_mode
        self.offload_to_cpu = offload_to_cpu
        self.release_inactive_state = release_inactive_state

        # Keep original requires_grad status of all parameters passed in.
        # We only BAdam-optimize parameters that were originally trainable.
        self.named_parameters = list(named_parameters)
        self.original_requires_grad = {p: p.requires_grad for name, p in self.named_parameters}
        trainable_named_params = [(name, p) for name, p in self.named_parameters if p.requires_grad]

        if not trainable_named_params:
            raise ValueError("No trainable parameters (requires_grad=True) found in named_parameters.")

        # Record parameter names for debugging and logging
        self.param_to_name = {p: name for name, p in self.named_parameters}

        # Partitioning
        if block_strategy == "transformer_layer":
            self.blocks = BlockPartitioner.partition_by_transformer_layer(trainable_named_params)
        elif block_strategy == "regex":
            self.blocks = BlockPartitioner.partition_by_regex(trainable_named_params, regex_patterns)
        elif block_strategy == "param_count":
            self.blocks = BlockPartitioner.partition_by_param_count(trainable_named_params, target_param_count)
        elif block_strategy == "custom":
            self.blocks = BlockPartitioner.partition_by_custom(trainable_named_params, custom_blocks)
        else:
            raise ValueError(f"Unknown block_strategy: {block_strategy}")

        logger.info(f"Initialized BAdam with {len(self.blocks)} parameter blocks using strategy '{block_strategy}'.")

        # Copy original param_groups structure to master copy.
        self.master_param_groups = []
        for group in base_optimizer.param_groups:
            group_copy = {k: v for k, v in group.items() if k != 'params'}
            group_copy['params'] = list(group['params'])
            self.master_param_groups.append(group_copy)

        # Set up properties needed by Optimizer subclassing
        self.param_groups = self.master_param_groups
        self.defaults = base_optimizer.defaults

        # Internal state caches
        self.cached_states = {}
        self.state = CombinedState(self.base_optimizer.state, self.cached_states)

        # Execution tracking state
        self.active_blocks_count = int(active_blocks_count)
        self.current_block_idx = 0
        self.current_step = 0
        self.block_indices_pool = []
        self.active_block_indices = []

        # Activate the first block
        self.activate_block(self.current_block_idx)

    @property
    def active_block_ratio(self) -> float:
        """
        Retorna a posição relativa do bloco ativo como um float entre 0.0 (camada inicial) e 1.0 (camada final).
        """
        if not hasattr(self, "blocks") or len(self.blocks) <= 1:
            return 0.0
        return float(self.current_block_idx) / float(len(self.blocks) - 1)

    def deactivate_all(self):
        """
        Deactivates all blocks: sets requires_grad=False, clears grads, and offloads optimizer states.
        """
        for block in self.blocks:
            for p in block:
                # 1. Cache/Offload optimizer state if active
                if p in self.base_optimizer.state:
                    state = self.base_optimizer.state[p]
                    if self.release_inactive_state:
                        # Release optimizer state completely for inactive blocks
                        pass
                    else:
                        # Move state tensors to CPU if offloading is enabled
                        if self.offload_to_cpu:
                            state_cpu = {}
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state_cpu[k] = v.cpu()
                                else:
                                    state_cpu[k] = v
                            self.cached_states[p] = state_cpu
                        else:
                            self.cached_states[p] = state

                    del self.base_optimizer.state[p]

                # 2. Deactivate requires_grad and clear gradient tensor to free memory
                p.requires_grad = False
                p.grad = None

    def activate_block(self, idx, active_indices=None):
        """
        Activates the selected block(s) by index: sets requires_grad=True, restores state, and maps parameter groups.
        """
        # Deactivate current active parameters first
        self.deactivate_all()

        self.current_block_idx = idx
        if active_indices is None:
            active_indices = [(idx + i) % len(self.blocks) for i in range(self.active_blocks_count)]
        self.active_block_indices = active_indices

        active_params = set()
        for b_idx in self.active_block_indices:
            active_params.update(self.blocks[b_idx])

        # 1. Enable requires_grad for the active block parameters
        for p in active_params:
            if self.original_requires_grad[p]:
                p.requires_grad = True

        # 2. Rebuild base_optimizer.param_groups containing only active parameters
        base_optimizer_groups = []
        for master_group in self.master_param_groups:
            active_params_in_group = [p for p in master_group['params'] if p in active_params]
            if active_params_in_group:
                group_copy = {k: v for k, v in master_group.items() if k != 'params'}
                group_copy['params'] = active_params_in_group
                base_optimizer_groups.append(group_copy)

        # Clear and update in-place to preserve optimizer reference
        self.base_optimizer.param_groups.clear()
        self.base_optimizer.param_groups.extend(base_optimizer_groups)

        # 3. Restore optimizer state from cache
        for p in active_params:
            if p in self.cached_states:
                state = self.cached_states[p]
                if self.offload_to_cpu:
                    state_gpu = {}
                    try:
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state_gpu[k] = v.to(p.device)
                            else:
                                state_gpu[k] = v
                    except Exception as e:
                        logger.warning(
                            f"Failed to move optimizer state to GPU for param "
                            f"{self.param_to_name.get(p, 'unknown')}: {e}. "
                            f"State will be re-initialized on next step."
                        )
                        state_gpu = {}

                    # Validate that the restored state is complete.
                    # bitsandbytes AdamW8bit requires 'state1'/'state2'; standard AdamW
                    # requires 'exp_avg'/'exp_avg_sq'. If the key set doesn't look right,
                    # clear it so the optimizer re-initializes cleanly on the first step
                    # rather than crashing with KeyError inside update_step.
                    if state_gpu:
                        has_bnb  = 'state1' in state_gpu
                        has_adam = 'exp_avg' in state_gpu
                        has_apollo = 'exp_avg' in state_gpu or 'projector' in state_gpu or 'seed' in state_gpu
                        if not (has_bnb or has_adam or has_apollo):
                            logger.warning(
                                f"Optimizer state for {self.param_to_name.get(p, 'unknown')} "
                                f"has unexpected keys {list(state_gpu.keys())}. "
                                f"Discarding to allow clean re-initialization."
                            )
                            state_gpu = {}

                    self.base_optimizer.state[p] = state_gpu
                    del self.cached_states[p]
                else:
                    # Validate non-offloaded state as well
                    if state:
                        has_bnb  = 'state1' in state
                        has_adam = 'exp_avg' in state
                        has_apollo = 'exp_avg' in state or 'projector' in state or 'seed' in state
                        if not (has_bnb or has_adam or has_apollo):
                            logger.warning(
                                f"Optimizer state for {self.param_to_name.get(p, 'unknown')} "
                                f"has unexpected keys {list(state.keys())}. "
                                f"Discarding to allow clean re-initialization."
                            )
                            state = {}
                    self.base_optimizer.state[p] = state
                    del self.cached_states[p]


        # Logging diagnostics
        active_names = []
        for b_idx in self.active_block_indices:
            active_names.extend([self.param_to_name.get(p, "unknown") for p in self.blocks[b_idx]])
        common_prefix = self._get_common_prefix(active_names)
        trainable_count = sum(sum(p.numel() for p in self.blocks[b_idx]) for b_idx in self.active_block_indices)
        gpu_mem, cpu_mem = self.get_optimizer_memory_estimate()

        logger.info(
            f"Activated Blocks {self.active_block_indices} | Name/Prefix: '{common_prefix}' | "
            f"Trainable Params: {trainable_count / 1e6:.2f}M | "
            f"State Memory Estimate - GPU: {gpu_mem / 1e6:.2f}MB, CPU: {cpu_mem / 1e6:.2f}MB"
        )

    def _get_common_prefix(self, names):
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        s1, s2 = min(names), max(names)
        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i]
        return s1

    def next_block(self):
        """
        Transitions to the next block(s) based on switch_mode.
        """
        num_blocks = len(self.blocks)
        if self.switch_mode == "ascending":
            next_idx = (self.current_block_idx + self.active_blocks_count) % num_blocks
            active_indices = [(next_idx + i) % num_blocks for i in range(self.active_blocks_count)]
        elif self.switch_mode == "descending":
            next_idx = (self.current_block_idx - self.active_blocks_count) % num_blocks
            active_indices = [(next_idx + i) % num_blocks for i in range(self.active_blocks_count)]
        elif self.switch_mode == "random":
            if not self.block_indices_pool:
                self.block_indices_pool = list(range(num_blocks))
                random.shuffle(self.block_indices_pool)
            active_indices = []
            for _ in range(self.active_blocks_count):
                if not self.block_indices_pool:
                    self.block_indices_pool = list(range(num_blocks))
                    random.shuffle(self.block_indices_pool)
                active_indices.append(self.block_indices_pool.pop(0))
            next_idx = active_indices[0]
        else:
            raise ValueError(f"Unknown switch_mode: {self.switch_mode}")

        logger.info(f"Switch Event: Blocks {self.active_block_indices} -> Blocks {active_indices}")
        self.activate_block(next_idx, active_indices=active_indices)

    def step(self, closure=None):
        """
        Synchronizes current hyperparameters from master_param_groups to the base optimizer's active groups,
        executes base_optimizer.step(), increments steps, and switches blocks when needed.
        """
        # Synchronize hyperparameters from self.master_param_groups to base_optimizer.param_groups
        for base_group in self.base_optimizer.param_groups:
            if not base_group['params']:
                continue
            first_param = base_group['params'][0]
            # Align base_group with its source master_group containing the same parameter
            for master_group in self.master_param_groups:
                if any(first_param is p for p in master_group['params']):
                    for k, v in master_group.items():
                        if k != 'params':
                            base_group[k] = v
                    break

        # Run step on active parameters only
        loss = self.base_optimizer.step(closure=closure)

        self.current_step += 1
        if self.current_step >= self.switch_block_every:
            self.current_step = 0
            self.next_block()

        return loss

    def zero_grad(self, set_to_none=True):
        """
        Zeros gradients of active parameter groups.
        """
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        Returns full state dict. Delegates optimizer state tensor serialization to the
        base_optimizer (which handles bitsandbytes 8-bit format, APOLLO projectors, etc.)
        and appends BAdam scheduling metadata.
        """
        # Before serializing, make sure active block state is in base_optimizer.state
        # (it should already be there during training, this is a safety check)
        # Then get the base optimizer's own state dict which handles its format correctly.

        # Temporarily restore active states to base_optimizer.state for serialization
        # (in case any were moved to cached_states during a deactivate_all call)
        active_params = set()
        for b_idx in self.active_block_indices:
            active_params.update(self.blocks[b_idx])
        for p in active_params:
            if p in self.cached_states and p not in self.base_optimizer.state:
                self.base_optimizer.state[p] = self.cached_states[p]

        # Also merge cached (inactive) states temporarily into base_optimizer.state
        # so the base optimizer's state_dict() captures everything
        temporarily_added = {}
        for p, state in self.cached_states.items():
            if p not in active_params:
                self.base_optimizer.state[p] = state
                temporarily_added[p] = state

        # Rebuild base_optimizer.param_groups to include ALL params (not just active block)
        # so the base optimizer's state_dict() can build correct param index mappings
        self.base_optimizer.param_groups.clear()
        for master_group in self.master_param_groups:
            group_copy = {k: v for k, v in master_group.items()}
            self.base_optimizer.param_groups.append(group_copy)

        # Get the full state dict from the base optimizer
        base_sd = self.base_optimizer.state_dict()

        # Clean up: remove temporarily added inactive states
        for p in temporarily_added:
            if p in self.base_optimizer.state:
                del self.base_optimizer.state[p]

        # Restore param_groups to only active block
        self.base_optimizer.param_groups.clear()
        base_optimizer_groups = []
        for master_group in self.master_param_groups:
            active_in_group = [p for p in master_group['params'] if p in active_params]
            if active_in_group:
                group_copy = {k: v for k, v in master_group.items() if k != 'params'}
                group_copy['params'] = active_in_group
                base_optimizer_groups.append(group_copy)
        self.base_optimizer.param_groups.extend(base_optimizer_groups)

        return {
            'base_optimizer_state': base_sd,
            'current_block_idx': self.current_block_idx,
            'active_block_indices': self.active_block_indices,
            'current_step': self.current_step,
            'block_indices_pool': self.block_indices_pool,
        }

    def load_state_dict(self, state_dict):
        """
        Restores full state dict: delegates optimizer state to base_optimizer.load_state_dict()
        (which handles bitsandbytes 8-bit tensors, APOLLO projectors, etc. correctly) and
        restores BAdam scheduling metadata.
        """
        current_block_idx = state_dict.get('current_block_idx', 0)
        active_block_indices = state_dict.get('active_block_indices', None)
        current_step = state_dict.get('current_step', 0)
        block_indices_pool = state_dict.get('block_indices_pool', [])

        # Boundary check to prevent IndexError if model configuration or blocks changed
        if current_block_idx >= len(self.blocks) or current_block_idx < 0:
            logger.warning(
                f"BAdam: Saved block index {current_block_idx} is out of bounds for the "
                f"current configuration of {len(self.blocks)} blocks. Resetting active block index to 0."
            )
            current_block_idx = 0
            active_block_indices = None

        if active_block_indices is not None:
            active_block_indices = [idx for idx in active_block_indices if 0 <= idx < len(self.blocks)]
            if len(active_block_indices) != self.active_blocks_count:
                logger.info(
                    f"BAdam: active_blocks_count changed from {len(active_block_indices)} (loaded) "
                    f"to {self.active_blocks_count} (configured). Regenerating active block indices."
                )
                active_block_indices = None
            elif not active_block_indices:
                active_block_indices = None

        if active_block_indices is None:
            active_block_indices = [(current_block_idx + i) % len(self.blocks) for i in range(self.active_blocks_count)]

        self.current_step = current_step
        self.block_indices_pool = block_indices_pool
        self.active_block_indices = active_block_indices

        base_sd = state_dict.get('base_optimizer_state')

        if base_sd is not None:
            # Temporarily set base_optimizer.param_groups to ALL params so the index
            # mapping matches what was saved in state_dict()
            self.base_optimizer.param_groups.clear()
            for master_group in self.master_param_groups:
                group_copy = {k: v for k, v in master_group.items()}
                self.base_optimizer.param_groups.append(group_copy)

            try:
                self.base_optimizer.load_state_dict(base_sd)
                logger.info("BAdam: base optimizer state loaded successfully via native load_state_dict.")
            except Exception as e:
                logger.warning(
                    f"BAdam: failed to load base optimizer state ({e}). "
                    f"Optimizer state will be re-initialized from scratch."
                )
                self.base_optimizer.state.clear()

            # Now move all states into cached_states (base optimizer has everything loaded)
            self.cached_states.clear()

            all_active = set()
            for b_idx in self.active_block_indices:
                all_active.update(self.blocks[b_idx])

            for p, state in list(self.base_optimizer.state.items()):
                if p not in all_active:
                    if self.offload_to_cpu:
                        state_cpu = {}
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state_cpu[k] = v.cpu()
                            else:
                                state_cpu[k] = v
                        self.cached_states[p] = state_cpu
                    else:
                        self.cached_states[p] = state
                    del self.base_optimizer.state[p]
        else:
            # Legacy format (saved by old BAdam code): fall back to old loading logic
            logger.warning(
                "BAdam: checkpoint uses legacy state format (no 'base_optimizer_state' key). "
                "Attempting legacy load..."
            )
            saved_states = state_dict.get('state', {})
            param_to_id = {}
            id_counter = 0
            all_params_list = []
            for group in self.master_param_groups:
                for p in group['params']:
                    param_to_id[p] = id_counter
                    all_params_list.append(p)
                    id_counter += 1

            self.base_optimizer.state.clear()
            self.cached_states.clear()

            for p_idx_str, state in saved_states.items():
                p_idx = int(p_idx_str)
                if p_idx < len(all_params_list):
                    p = all_params_list[p_idx]
                    state_copy = {k: v for k, v in state.items()}
                    if self.offload_to_cpu:
                        for k, v in list(state_copy.items()):
                            if isinstance(v, torch.Tensor):
                                state_copy[k] = v.cpu()
                    self.cached_states[p] = state_copy

        # Activate the designated block (restores state for active block from cache)
        self.activate_block(current_block_idx, active_indices=self.active_block_indices)


    def get_optimizer_memory_estimate(self):
        """
        Returns a tuple (gpu_bytes, cpu_bytes) indicating the estimated footprint of active/cached states.
        """
        gpu_bytes = 0
        cpu_bytes = 0

        # Helper to compute size of state dictionaries
        def accumulate_sizes(state_dict):
            g, c = 0, 0
            for p, state in state_dict.items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        bytes_size = v.numel() * v.element_size()
                        if v.device.type == 'cuda':
                            g += bytes_size
                        else:
                            c += bytes_size
            return g, c

        g_active, c_active = accumulate_sizes(self.base_optimizer.state)
        g_cached, c_cached = accumulate_sizes(self.cached_states)

        gpu_bytes = g_active + g_cached
        cpu_bytes = c_active + c_cached

        return gpu_bytes, cpu_bytes

    def __getattr__(self, name):
        if name == "base_optimizer":
            raise AttributeError("base_optimizer is not initialized")
        return getattr(self.base_optimizer, name)
