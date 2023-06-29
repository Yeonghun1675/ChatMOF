from typing import Any, Dict
from pathlib import Path
from pydantic import BaseModel, root_validator
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from chatmof import __root_dir__
from chatmof.tools.genetic_algorithm.base import Generator


def _get_generator(
        verbose: bool = False,
        **kwargs: Any
) -> BaseTool:
    return Tool(
        name="generator",
        description=(
            "Generate material for specific properties using genetic algorithm. "            
        ),
        func=Generator(verbose=verbose).run,
    )