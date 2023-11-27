import os
from typing import Any
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

from chatmof.config import config
from chatmof.tools.ase_repl.base import ASETool


def _get_ase_repl(
        llm: BaseLanguageModel,
        file_path: str = config['lookup_dir'],
        verbose: bool = False,
        **kwargs: Any) -> BaseTool:

    return Tool(
        name="ase_repl",
        description=(
            "A tools for analyzing materials using the ASE library."
            "It allows you to perform tasks using ASE, such as atom coordinates and cell information"
            "The input must be a detailed full sentence to answer the question."
        ),
        func=ASETool.from_llm(
            llm=llm, verbose=verbose,
        ).run
    )
