from typing import Any, Dict
from pathlib import Path
from pydantic import BaseModel, root_validator
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from chatmof import __root_dir__
from chatmof.tools.genetic_algorithm.base import Generator


def _get_generator(
        llm: BaseLanguageModel,
        verbose: bool = False,
        **kwargs: Any
) -> BaseTool:
    return Tool(
        name="generator",
        description=(
            "The tool to use when you need to create materials with specific properties. "
            "`generator` must only be used when the original question wants to generate. "
            "input must be provided in the form of a full sentence. "
        ),
        func=Generator.from_llm(llm=llm, verbose=verbose).run,
    )