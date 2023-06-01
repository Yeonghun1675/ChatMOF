from typing import Any, Dict
from pydantic import BaseModel, root_validator
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
#from chatmof import __root_dir__
from chatmof.tools.predictor.predict_property import Predictor


def _get_predict_properties(
        verbose: bool = False,
        **kwargs: Any
) -> BaseTool:
    return Tool(
        name="predictor",
        description=(
            #"Before using the predictor, you must check that the search_csv contains no information. "
            "Predictor is a tool that uses machine learning to predict the properties of a structure. "
            "To use a predictor, you need two inputs seperated by \",\". "
            "\"property\" is the single target you want to predict. "
            "list of properties : bandgap, hydrogen_uptake, hydrogen_diffusivity"
            "\"data_list\" is the name of the structure you want to predict. "
            "If there are multiple structures, they are separated by \",\". "
            "If you want to get properties for all structures, write \"all\" instead of a structure name."
            #"(The structure exists in ~, you need to make sure the cif file exists.) "
            "\n"
            "Example of predictor : bandgap,structure_A,structure_B"
        ),
        func=Predictor(verbose=verbose).run,
    )