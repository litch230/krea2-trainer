from accelerate import init_empty_weights
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from library.krea2.krea2_encoder import QWEN3_VL_4B_INSTRUCT_CONFIG


def test_qwen3_vl_4b_attention_shapes_match_checkpoint():
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    attention = model.model.language_model.layers[0].self_attn
    assert tuple(attention.q_proj.weight.shape) == (4096, 2560)
    assert tuple(attention.k_proj.weight.shape) == (1024, 2560)
    assert tuple(attention.v_proj.weight.shape) == (1024, 2560)
    assert tuple(attention.o_proj.weight.shape) == (2560, 4096)
