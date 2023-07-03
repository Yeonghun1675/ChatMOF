import os
import re
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.tools.python.tool import PythonAstREPLTool
import tiktoken

from chatmof.config import config
from chatmof.tools.search_csv.prompt import DF_PROMPT


class TableSearcher(Chain):
    """Tools that search csv using Pandas agent"""
    llm_chain: LLMChain
    df: pd.DataFrame
    encoder: tiktoken.core.Encoding
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _parse_output(self, text:str) -> Dict[str, Any]:
        thought = re.search(r"(?<!Final )Thought:\s*(.+?)\s*(\n|$)", text, re.DOTALL)
        input_ = re.search(r"Input:\s*(?:```|`)(.+?)(?:```|`)\s*(\n|$)", text, re.DOTALL)
        final_thought = re.search(r"Final Thought:\s*(.+?)\s*(\n|$)", text, re.DOTALL)
        final_answer = re.search(r"Final Answer:\s*(.+?)\s*(\n|$)", text, re.DOTALL)

        if (not input_) and (not final_answer):
            raise ValueError(f'unknown format from LLM: {text}')
        
        return {
            'Thought': (thought.group(1) if thought else None),
            'Input': (input_.group(1).strip() if input_ else None),
            'Final Thought' : (final_thought.group(1) if final_thought else None),
            'Final Answer': (final_answer.group(1) if final_answer else None),
        }
    
    def _clear_name(self, text:str) -> str:
        remove_list = ['_clean_h', '_clean', '_charged', '_manual', '_ion_b', '_auto', '_SL', ]
        str_remove_list = r"|".join(remove_list)
        return re.sub(rf"({str_remove_list})", "", text)
    
    @staticmethod
    def _get_df(file_path: str):
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f'table must be .csv, .xlsx, or .json, not {file_path.suffix}')

        return df

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        return_observation = inputs.get('return_observation', False)
        
        agent_scratchpad = ''
        max_iteration = config['max_iteration']

        input_ = self._clear_name(inputs[self.input_key])

        for i in range(max_iteration + 1):
            llm_output = self.llm_chain.run(
                df_head = self.df.head().to_markdown(),
                question=input_,
                agent_scratchpad = agent_scratchpad,
                callbacks=callbacks,
                stop=['Observation:', 'Question:',]
            )
            output = self._parse_output(llm_output)

            if output['Final Answer']:
                if output['Thought']:
                    raise ValueError(llm_output)
                
                run_manager.on_text(f"\n[Table Searcher] Final Thought: ", verbose=self.verbose)
                run_manager.on_text(output['Final Thought'], verbose=self.verbose, color='yellow')

                final_answer = output['Final Answer']

                check_sentence = ' You must make sure that this answer leads to the final answer. You should not do any additional verification on the answer.'
                if re.search(r'nothing', final_answer):
                    final_answer = 'There are no data in database.' # please use tool `predictor` to get answer.'
                elif '.' in final_answer:
                    final_answer += check_sentence
                else:
                    final_answer = f'The answer is {final_answer}.{check_sentence}'
                return {self.output_key: final_answer}
            
            elif i >= max_iteration:
                run_manager.on_text(f"\n[Table Searcher] Final Thought: ", verbose=self.verbose)
                run_manager.on_text('There are no data in database', verbose=self.verbose, color='yellow')
                final_answer = 'There are no data in database.' # please use tool `predictor` to get answer.'
                return {self.output_key: final_answer}

            else:
                run_manager.on_text(f"\n[Table Searcher] Thought: ", verbose=self.verbose)
                run_manager.on_text(output['Thought'], verbose=self.verbose, color='yellow')
                run_manager.on_text(f"\n[Table Searcher] Input: \n", verbose=self.verbose)
                run_manager.on_text(output['Input'], verbose=self.verbose, color='yellow')

            pytool = PythonAstREPLTool(locals={'df':self.df})
            observation = str(pytool.run(output['Input'])).strip()

            num_tokens = len(self.encoder.encode(observation))
            if num_tokens > 3400:
                observation = "The number of tokens has been exceeded. To reduce the length of the message, please modify code to only pull up to 200 data."
            elif return_observation:
                run_manager.on_text(f"\n[Table Searcher] Observation: ", verbose=self.verbose)
                if "\n" in observation:
                    run_manager.on_text("\n"+observation, verbose=self.verbose, color='yellow')
                else:
                    run_manager.on_text(observation, verbose=self.verbose, color='yellow')
                return {self.output_key: observation}

            run_manager.on_text(f"\n[Table Searcher] Observation: ", verbose=self.verbose)
            if "\n" in observation:
                run_manager.on_text("\n"+observation, verbose=self.verbose, color='yellow')
            else:
                run_manager.on_text(observation, verbose=self.verbose, color='yellow')

            agent_scratchpad += 'Thought: {}\nInput: {} \nObservation: {}\n'\
                .format(output['Thought'], output['Input'], observation)

        raise AssertionError('Code Error! please report to author!')

    @classmethod
    def from_filepath(
        cls,
        llm: BaseLanguageModel,
        file_path: Path = Path(config['lookup_dir']),
        prompt: str = DF_PROMPT,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['df_head', 'question', 'agent_scratchpad']
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        df = cls._get_df(file_path)
        encoder = tiktoken.encoding_for_model(llm.model_name)
        return cls(llm_chain=llm_chain, df=df, encoder=encoder, **kwargs)
    
    @classmethod
    def from_dataframe(
        cls,
        llm: BaseLanguageModel,
        dataframe: pd.DataFrame,
        prompt: str = DF_PROMPT,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['df_head', 'question', 'agent_scratchpad']
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        encoder = tiktoken.encoding_for_model(llm.model_name)
        return cls(llm_chain=llm_chain, df=dataframe, encoder=encoder, **kwargs)