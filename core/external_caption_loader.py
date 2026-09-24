import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_caption_file(filepath: str) -> Optional[str]:
    """
    Reads a caption file safely, handling encoding errors.
    Returns the loaded string stripped, or None if error.
    """
    if not filepath or not os.path.isfile(filepath):
        return None
        
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read().strip()
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"[Caption Loader] Error reading {filepath}: {e}")
            break
            
    logger.error(f"[Caption Loader] Failed to decode caption file {filepath} with supported encodings.")
    return None
