from typing import Any, Dict
from pathlib import Path
from pydantic import BaseModel, root_validator
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from chatmof import __root_dir__
from chatmof.tools.predictor.base import Predictor



def _get_predict_properties(
        verbose: bool = False,
        **kwargs: Any
) -> BaseTool:
    return Tool(
        name="predictor",
        description=(
            "Predict material properties using machine learning. "
            "More imprecise than the search_csv tool, but can be used universally."
        ),
        func=Predictor(verbose=verbose).run,
    )