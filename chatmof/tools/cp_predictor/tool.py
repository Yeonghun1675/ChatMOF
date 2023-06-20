from typing import Any, Dict
from pathlib import Path
from pydantic import BaseModel, root_validator
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from chatmof import __root_dir__
from chatmof.tools.predictor.runner import Predictor



def _get_predict_properties(
        verbose: bool = False,
        **kwargs: Any
) -> BaseTool:
    return Tool(
        name="predictor",
        description=(
            "Before using the predictor tool, ensure that 'search_csv' is empty; "
            "the predictor employs machine learning to estimate the attributes of structures and requires two comma-separated inputs - 'property', "
            "representing the specific characteristic to be predicted, and 'data_list' denoting the structure(s) in question "
            "(use a comma for multiple structures, or “all” for every structure); "
            "for instance, 'bandgap,structure_A,structure_B' would predict the bandgap for structure_A and structure_B. "
            "A predictor must have only one property at a time. "
            f"Property to take, must be one of [{_str_predictable_properties}]"
        ),
        func=Predictor(verbose=verbose).run,
    )