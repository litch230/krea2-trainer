import os
import logging
from typing import Optional

from utils.path_resolver import resolve_sibling_caption_paths
from core.external_caption_loader import load_caption_file
from core.caption_merger import merge_tags_and_nl_captions

logger = logging.getLogger(__name__)

class CaptionResolver:
    def __init__(
        self,
        use_external_caption_folders: bool = False,
        tags_folder_name: str = "tags",
        nl_folder_name: str = "NL",
        merge_tags_and_nl: bool = True,
        fallback_to_original_caption: bool = True,
        use_only_tags: bool = False,
        use_only_nl: bool = False,
        warn_on_missing_captions: bool = False,
        ignore_missing_captions: bool = True,
    ):
        self.use_external_caption_folders = use_external_caption_folders
        self.tags_folder_name = tags_folder_name
        self.nl_folder_name = nl_folder_name
        self.merge_tags_and_nl = merge_tags_and_nl
        self.fallback_to_original_caption = fallback_to_original_caption
        self.use_only_tags = use_only_tags
        self.use_only_nl = use_only_nl
        self.warn_on_missing_captions = warn_on_missing_captions
        self.ignore_missing_captions = ignore_missing_captions

    def resolve_caption(self, img_path: str, base_dir: str, original_caption: Optional[str] = None) -> Optional[str]:
        """
        Resolves the final caption for a given image.
        """
        if not self.use_external_caption_folders:
            return original_caption
            
        sibling_paths = resolve_sibling_caption_paths(
            img_path, base_dir, self.tags_folder_name, self.nl_folder_name
        )
        
        tags_content = None
        nl_content = None
        
        if sibling_paths["tags"] and not self.use_only_nl:
            tags_content = load_caption_file(sibling_paths["tags"])
            
        if sibling_paths["NL"] and not self.use_only_tags:
            nl_content = load_caption_file(sibling_paths["NL"])
            
        logger.debug(
            f"[Caption Loader]\n"
            f"Image: {img_path}\n"
            f"Loaded tags: {sibling_paths['tags']} (found: {tags_content is not None})\n"
            f"Loaded NL: {sibling_paths['NL']} (found: {nl_content is not None})"
        )
        
        final_caption = None
        if self.merge_tags_and_nl:
            final_caption = merge_tags_and_nl_captions(tags_content, nl_content)
        else:
            if tags_content:
                final_caption = tags_content
            elif nl_content:
                final_caption = nl_content
                
        if not final_caption:
            if self.fallback_to_original_caption:
                final_caption = original_caption
                
        if not final_caption or final_caption.strip() == "":
            if self.warn_on_missing_captions:
                logger.warning(f"[Caption Loader] Missing caption for image: {img_path}")
        else:
            logger.debug(
                f"[Caption Loader] Image: {img_path} | Final merged caption length: {len(final_caption)} chars"
            )
            
        return final_caption
