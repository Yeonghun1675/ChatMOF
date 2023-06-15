import os
import re
import json
from pydantic import BaseModel
from langchain.agents import create_csv_agent
from langchain.prompts import PromptTemplate, BasePromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.schema import OutputParserException
from langchain.tools import BaseTool

from chatmof import __root_dir__
from chatmof.utils import preprocess_json_input
from chatmof.tools.search_csv.prompt import PROMPT


class CSVSearchAgent(BaseModel):
    """Tool that search data in CSV files.
    """
    
    llm: BaseLanguageModel
    file_path: str|None = os.path.join(__root_dir__, 'database/tables/coremof.csv'),    
    prompt: BasePromptTemplate = PROMPT
    verbose: bool = False

    def run(
            self,       
            query: str
    ) -> str:
        """Running csv_agent and parse result"""

        if re.search(r"\b(all|every)\b", query):
            return (
                'The quary for search_csv must be written to produce a answer with the fewest number of tokens. '
                'You must write your query to extract only the information you need instead of getting all the information. '
                'For instance, rather than querying "What is the pore volume of all structures?", you can write a query that asks "What are the top 10 structures with the smallest pore volume?". '
                'Please rewrite the query for tool "search_csv".'
            )
        
        prompt_template = PromptTemplate.from_template(
            template=PROMPT,
            template_format='jinja2',
        )

        agent = create_csv_agent(
            self.llm, 
            self.file_path, 
            verbose=self.verbose,   
            max_iterations=2,
        )
        template = prompt_template.format(question=query, filename=self.file_path)

        try:
            result = agent.run(template) 
        except StopIteration:
            return "Failed to parse response from agent"
        except OutputParserException as e:
            text = str(e)
            if text.startswith("Could not parse LLM output: `"):
                text = text.removeprefix("Could not parse LLM output: `").removesuffix("`")

            return (
                f"Result : {str(e)}\n"
                "Please check that these result lead to a final answer."
            )
        
        return self.parse_data(result)
            
    def parse_data(self, result):
        try:
            data = json.loads(result, strict=False)
            if isinstance(data, list):
                d_tmp = data[0]
            elif isinstance(data, dict):
                d_tmp = data
            else:
                raise TypeError(f'result must be list, or dict, not {type(data)}')
            if d_tmp['property'] is None or d_tmp['name'] is None:
                return f"Result : There are no data in database."
            else:
                return f"Result : {json.dumps(data)}\nPlease check that these result lead to a final answer."
        except json.JSONDecodeError:
            return f"Result : {result}\nPlease check that these result lead to a final answer."
        
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        file_path: str|None = os.path.join(__root_dir__, 'database/tables/coremof.csv'),    
        verbose: bool = False
    ) -> BaseModel:
        return cls(llm=llm, file_path=file_path, verbose=verbose)
    