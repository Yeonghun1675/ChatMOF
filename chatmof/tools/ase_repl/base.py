import os
import re
from pathlib import Path
import pandas as pd
import ase
import ase.io
import ase.visualize
from typing import Dict, Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.tools.python.tool import PythonAstREPLTool
import tiktoken

from chatmof.config import config
from chatmof.tools.ase_repl.prompt import ASE_PROMPT


class ASETool(Chain):
    """Tools that search csv using ASE agent"""
    llm_chain: LLMChain
    atoms: ase.Atoms
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _parse_output(self, text:str) -> Dict[str, Any]:
        thought = re.search(r"(?<!Final )Thought:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        input_ = re.search(r"Input:\s*(?:```|`)?(.+?)(?:```|`)?\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        final_thought = re.search(r"Final Thought:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        final_answer = re.search(r"Final Answer:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|$)", text, re.DOTALL)
        
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
    
    def _write_log(self, action, text, run_manager):
        run_manager.on_text(f"\n[Table Searcher] {action}: ", verbose=self.verbose)
        run_manager.on_text(text, verbose=self.verbose, color='yellow')
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        return_observation = inputs.get('return_observation', False)
        
        agent_scratchpad = ''
        information = inputs.get('information', "If unit exists, you must include it in the final output.")
        max_iteration = config['max_iteration']

        input_ = self._clear_name(inputs[self.input_key])

        for i in range(max_iteration + 1):
            llm_output = self.llm_chain.run(
                df_index = str(list(self.df)),
                information = information,
                df_head = self.df.head().to_markdown(),
                question=input_,
                agent_scratchpad = agent_scratchpad,
                callbacks=callbacks,
                stop=['Observation:', 'Question:',]
            )

            if not llm_output.strip():
                agent_scratchpad += 'Thought: '
                llm_output = self.llm_chain.run(
                    df_index = str(list(self.df)),
                    information = information,
                    df_head = self.df.head().to_markdown(),
                    question=input_,
                    agent_scratchpad = agent_scratchpad,
                    callbacks=callbacks,
                    stop=['Observation:', 'Question:',]
                )
            
            #if llm_output.endswith('Final Answer: success'):
            if re.search(r'Final Answer: (success|.* above|.* DataFrames?).?$', llm_output):
                thought = f'Final Thought: we have to answer the question `{input_}` using observation\n'
                agent_scratchpad += thought
                llm_output = self.llm_chain.run(
                    df_index = str(list(self.df)),
                    df_head = self.df.head().to_markdown(),
                    question=input_,
                    information = information,
                    agent_scratchpad = agent_scratchpad,
                    callbacks=callbacks,
                    stop=['Observation:', 'Question:',]
                )
                llm_output = thought + llm_output

            output = self._parse_output(llm_output)

            if output['Final Answer']:
                if output['Thought']:
                    raise ValueError(llm_output)
                
                self._write_log('Final Thought', output['Final Thought'], run_manager)

                final_answer: str = output['Final Answer']
                #if final_answer == 'success':
                #    agent_scratchpad += 'Thought: {}\nInput: {} \nObservation: {}\n'\
                #            .format(output['Thought'], output['Input'], observation)

                #check_sentence = ' Check to see if this answer can be your final answer, and if so, you should submit your final answer. You should not do any additional verification on the answer.'
                check_sentence = ''
                if re.search(r'nothing', final_answer):
                    final_answer = 'There are no data in database.' # please use tool `predictor` to get answer.'
                elif final_answer.endswith('.'):    
                    final_answer += check_sentence
                else:
                    final_answer = f'The answer for question "{input_}" is {final_answer}.{check_sentence}'
                return {self.output_key: final_answer}
            
            elif i >= max_iteration:
                final_answer = 'There are no data in database'
                self._write_log('Final Thought', final_answer, run_manager)
                return {self.output_key: final_answer}

            else:
                self._write_log('Thought', output['Thought'], run_manager)
                self._write_log('Input', output['Input'], run_manager)
                
            pytool = PythonAstREPLTool(locals={'df':self.df})
            observation = str(pytool.run(output['Input'])).strip()

            num_tokens = len(self.encoder.encode(observation))
            
            if return_observation:
                if "\n" in observation:
                    self._write_log('Observation', "\n"+observation, run_manager)
                else:
                    self._write_log('Observation', observation, run_manager)
                return {self.output_key: observation}
            
            if num_tokens > 3400:
                raise ValueError('The number of tokens has been exceeded.')
                observation = f"The number of tokens has been exceeded. To reduce the length of the message, please modify code to only pull up to {self.num_max_data} data."
                self.num_max_data = self.num_max_data // 2

            if "\n" in observation:
                self._write_log('Observation', "\n"+observation, run_manager)
            else:
                self._write_log('Observation', observation, run_manager)

            agent_scratchpad += 'Thought: {}\nInput: {} \nObservation: {}\n'\
                .format(output['Thought'], output['Input'], observation)

        raise AssertionError('Code Error! please report to author!')

    @classmethod
    def from_filepath(
        cls,
        llm: BaseLanguageModel,
        file_path: Path = Path(config['lookup_dir']),
        prompt: str = ASE_PROMPT,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['df_index', 'information', 'df_head', 'question', 'agent_scratchpad']
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        df = cls._get_df(file_path)
        encoder = tiktoken.encoding_for_model(llm.model_name)
        return cls(llm_chain=llm_chain, df=df, encoder=encoder, **kwargs)
    
    @classmethod
    def from_atoms(
        cls,
        llm: BaseLanguageModel,
        atoms: ase.Atoms,
        prompt: str = ASE_PROMPT,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['df_index', 'information', 'df_head', 'question', 'agent_scratchpad']
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        encoder = tiktoken.encoding_for_model(llm.model_name)
        return cls(llm_chain=llm_chain, atoms=atoms, encoder=encoder, **kwargs)