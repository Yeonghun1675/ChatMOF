import re


def preprocess_json_input(input_str: str) -> str:
    # Replace single backslashes with double backslashes,
    # while leaving already escaped ones intact
    corrected_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str
    )
    return corrected_str