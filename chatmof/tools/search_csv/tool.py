import os
from typing import Any
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

from chatmof import __root_dir__
from chatmof.tools.search_csv.search import CSVSearchAgent


def _get_search_csv(
        llm: BaseLanguageModel,
        file_path: str|None = os.path.join(__root_dir__, 'database/tables/coremof.csv'),    
        verbose: bool = False,
        **kwargs: Any) -> BaseTool:
    return Tool(
        name="search_csv",
        description=(
                "A tools that extract information from a look-up table in the database. "
                "Takes a question and provides the answer in JSON format. \n"
                'The question must be written to produce a answer with the fewest number of tokens. '
                'For example, you must not write the query to find the surface area of all structures. \n'
                "Example of search_csv\n"
                "Query : What is the surface area of ABEYIR_clean?\n"
                'Result : ["name": "ABEYIR_clean", "property": "Accessible Surface Area", "value": 446.707, "unit": "m^2/g"]'
        ),
        func=CSVSearchAgent.from_llm(llm, file_path, verbose=verbose).run,
        coroutine=CSVSearchAgent.from_llm(llm, file_path, verbose=verbose).arun,
    )
