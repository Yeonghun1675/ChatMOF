from typing import Any
from pydantic import BaseModel, Field
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
            "Before using the predictor, you must check that the search_csv contains no information."
            
            "To use predictor, property and names of structures which want to get values are needed.\n"
            "Example of predictor\n"
            "data_list : \"structure_A,structure_B\""
            "property : \"bandgap\""
        ),

        func=Predictor(verbose=verbose).run,
        #coroutine=PredictMofT().arun
    )