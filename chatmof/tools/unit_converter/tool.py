import os
from typing import Any
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

from chatmof.config import config
from chatmof.tools.unit_converter.base import UnitConverter


def _get_unit_converter(
        llm: BaseLanguageModel,
        verbose: bool = False,
        **kwargs: Any) -> BaseTool:

    return Tool(
        name="unit_converter",
        description=(
            "A tool for converting unit. "
            "Ask a question about unit conversion and it will tell you how to convert and what information you need. "
            ""
        ),
        func=UnitConverter.from_llm(
            llm=llm, verbose=verbose
        ).run
    )
