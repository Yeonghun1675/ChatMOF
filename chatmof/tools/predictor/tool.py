from typing import Any, Dict
from pathlib import Path
from pydantic import BaseModel, root_validator
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from chatmof import __root_dir__
from chatmof.tools.predictor.base import Predictor



def _get_predict_properties(
        llm: BaseLanguageModel,
        verbose: bool = False,
        **kwargs: Any
) -> BaseTool:
    return Tool(
        name="predictor",
        description=(
            "Useful tool to predict material properties using machine learning. "
            "More imprecise than the search_csv tool, but can be used universally. "
            "You must use predictor before using google_search or wikipedia. "
            "The input must be a detailed full sentence to answer the question."
        ),
        func=Predictor.from_llm(llm=llm, verbose=verbose).run,
    )