import os
import pathlib
from typing import Optional, Dict

SUPPORTED_EXTENSIONS = [".txt", ".caption", ".md"]

def resolve_sibling_caption_paths(
    img_path: str,
    base_dir: str,
    tags_dir_name: str = "tags",
    nl_dir_name: str = "NL",
) -> Dict[str, Optional[str]]:
    """
    Given an image path and its subset base directory,
    resolves the corresponding sibling tags and NL caption paths.
    """
    img_path_obj = pathlib.Path(img_path)
    img_dir = img_path_obj.parent
    img_stem = img_path_obj.stem
    
    tags_dir = img_dir / tags_dir_name
    nl_dir = img_dir / nl_dir_name
    
    resolved = {
        "tags": None,
        "NL": None
    }
    
    # Resolve tags path
    if os.path.isdir(tags_dir):
        for ext in SUPPORTED_EXTENSIONS:
            cand = tags_dir / (img_stem + ext)
            if cand.is_file():
                resolved["tags"] = str(cand)
                break
                
    # Resolve NL path
    if os.path.isdir(nl_dir):
        for ext in SUPPORTED_EXTENSIONS:
            cand = nl_dir / (img_stem + ext)
            if cand.is_file():
                resolved["NL"] = str(cand)
                break
                
    return resolved
