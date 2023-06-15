from typing import Any, Dict
from pathlib import Path
from pydantic import BaseModel, root_validator
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from chatmof import __root_dir__
from chatmof.tools.predictor.predict_property import Predictor


_predictable_properties = [
    path.stem for path in (Path(__root_dir__)/'database/load_model').iterdir()
]
_str_predictable_properties = ",".join(_predictable_properties)


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
            #"predictor can predict only one property for each step. "
            f"Property to take, must be one of [{_str_predictable_properties}]"
            #"Before using the predictor, you must check that the search_csv contains no information. "
            #"Predictor is a tool that uses machine learning to predict the properties of a structure. "
            #"To use a predictor, you need two inputs seperated by \",\". "
            #"\"property\" is the single target you want to predict. "
            #"list of properties : bandgap, hydrogen_uptake, hydrogen_diffusivity"
            #"\"data_list\" is the name of the structure you want to predict. "
            #"If there are multiple structures, they are separated by \",\". "
            #"If you want to get properties for all structures, write \"all\" instead of a structure name. "
            #"\n"
            #f"predictable property : {_str_predictable_properties}"
            #"(The structure exists in ~, you need to make sure the cif file exists.) "
            #"\n"
            #"Example of predictor : bandgap,structure_A,structure_B"
        ),
        func=Predictor(verbose=verbose).run,
    )