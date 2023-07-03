import os
from typing import Any
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

from chatmof.config import config
from chatmof.tools.search_csv.base import TableSearcher


def _get_search_csv(
        llm: BaseLanguageModel,
        file_path: str = config['lookup_dir'],
        verbose: bool = False,
        **kwargs: Any) -> BaseTool:

    return Tool(
        name="search_csv",
        description=(
                "A tools that extract accurate properties from a look-up table in the database. "
                #"input must be provided in the form of a full sentence. "
                "The input must be a detailed full sentence to answer the question."
        ),
        func=TableSearcher.from_filepath(
            llm=llm, file_path=file_path, verbose=verbose
        ).run
    )
