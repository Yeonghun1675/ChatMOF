from pathlib import Path
from chatmof import __root_dir__


_predictable_properties = [
    path.stem for path in (Path(__root_dir__)/'database/load_model').iterdir()
]
_str_predictable_properties = ",".join(_predictable_properties)


PROMPT = (
"the predictor employs machine learning to estimate the attributes of structures and requires two comma-separated inputs - 'property', "
"representing the specific characteristic to be predicted, and 'data_list' denoting the structure(s) in question "
"(use a comma for multiple structures, or “all” for every structure); "
"for instance, 'bandgap,structure_A,structure_B' would predict the bandgap for structure_A and structure_B. "
"A predictor must have only one property at a time. "
f"Property to take, must be one of [{_str_predictable_properties}]"
)