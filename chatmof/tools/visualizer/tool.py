import os
from typing import Any
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

from chatmof.config import config
from chatmof.tools.visualizer.base import Visualizer


def _get_visualizer(
        llm: BaseLanguageModel,
        file_path: str = config['lookup_dir'],
        verbose: bool = False,
        **kwargs: Any) -> BaseTool:

    return Tool(
        name="visualizer",
        description=(
                "A tools that visualize structure. "
                "The input must be a detailed full sentence to answer the question."
        ),
        func=Visualizer.from_filepath(
            llm=llm, file_path=file_path, verbose=verbose
        ).run
    )
