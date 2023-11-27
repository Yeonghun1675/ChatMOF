import re
from typing import List
from pathlib import Path


def preprocess_json_input(input_str: str) -> str:
    # Replace single backslashes with double backslashes,
    # while leaving already escaped ones intact
    corrected_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str
    )
    return corrected_str


def search_file(name: str, direc: Path) -> List[Path]:
    if '*' in name:
        files = list(direc.glob(name))
        if files:
            return files
    else:
        f_name = direc/name
        if f_name.exists():
            return [f_name]
        
    iterdir = sorted([i for i in direc.iterdir() if i.is_dir()])
    for direc in iterdir:
        files = search_file(name, direc)
        if files:
            return files
            
    return []