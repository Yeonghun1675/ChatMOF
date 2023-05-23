import os
import json
from pydantic import BaseModel
from langchain.agents import create_csv_agent
from langchain.prompts import PromptTemplate, BasePromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool

from chatmof import __root_dir__
from chatmof.utils import preprocess_json_input


PROMPT = (
    'SYSTEM: Your task involves receiving a question from a HUMAN, then extracting the corresponding information from a CSV file in response to this query. '
    'The final result must be structured as a JSON object, containing "name", "property", "value", and "unit" as keys in the dictionary or within a list of dictionaries. '
    'The "unit" is the value inside parentheses in the CSV header, and it should be included without parentheses. '
    'If there is no value for a key, return None. '
    'If a KeyError or IndexError occurs, return {\"name\": null, \"property\": null, \"value\": null, \"unit\": null}. '
    'Path of CSV file is "{{filename}}", Ensure that the file name is correctly referenced in your code. '
    'If the dataframe doesn\'t contain any data for the requested property or requested structure in the Thought step, '
    'You must stop and directly provide the output {\"name\": null, \"property\": null, \"value\": null, \"unit\": null} without any additional steps. '
    'submit the output {\"name\": null, \"property\": null, \"value\": null, \"unit\": null} without further steps.'
    'If it was extracted in JSON format during the observation phase, it will be returned directly without further checking. \n'
    '\n'
    'Example of plan\n'
    'Action: python_repl_ast\n'
    'Action Input:\n'
    "import pandas as pd\n"
    "import json\n"
    "\n"
    'try:\n'
    '    df = pd.read_csv("filename.csv")\n'
    "    row = df.loc[df['name'] == 'ABEXIQ_clean']\n"
    "    value = row['Accessible Surface Area (m^2/g)'].values[0]\n"
    '    result = {\"name\": \"ABEXIQ_clean\", \"property\": \"Accessible Surface Area\", \"value\": value, \"unit\": \"m^2/g\"}\n'
    "except (KeyError, IndexError):\n"
    '    result = {\"name\": None, \"property\": None, \"value\": None, \"unit\": None}\n'
    'json.dumps(result)\n'
    '\n'
    'Result of python_repl_ast:\n'
    '{\"name\": \"ABEXIQ_clean\", \"property\": \"Accessible Surface Area\", \"value\": 534.065, \"unit\": \"m^2/g\"} # if data is existed\n'
    '{\"name\": null, \"property\": null, \"value\": null, \"unit\": null} # if data is not existed\n'
    '\n'
    'HUMAN: {{question}}'
)


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

        prompt_template = PromptTemplate.from_template(
            template=PROMPT,
            template_format='jinja2',
        )

        agent = create_csv_agent(self.llm, self.file_path, verbose=self.verbose)
        template = prompt_template.format(question=query, filename=self.file_path)
        try:
            result = agent.run(template) 
        except StopIteration:
            return "Failed to parse response from agent"
        
        try:
            data = json.loads(result, strict=False)
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(result)
            try:
                data = json.loads(preprocessed_text, strict=False)
            except Exception:
                return "Error : Parsing Error!"
        
        if data['property'] is None or data['name'] is None:
            return f"Result : There are no data in database."
        else:
            return f"Result : {json.dumps(data)}"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        file_path: str|None = os.path.join(__root_dir__, 'database/tables/coremof.csv'),    
        verbose: bool = False
    ) -> BaseModel:
        return cls(llm=llm, file_path=file_path)
    