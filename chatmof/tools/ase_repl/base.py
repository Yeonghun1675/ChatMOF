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

from chatmof.config import config
from chatmof.utils import search_file
from chatmof.tools.ase_repl.prompt import ASE_PROMPT


class ASETool(Chain):
    """Tools that search csv using ASE agent"""
    llm_chain: LLMChain
    data_dir: Path = Path(config['structure_dir'])
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _parse_output(self, text:str) -> Dict[str, Any]:
        thought = re.search(
            r"(?<!Final )Thought:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|Material|Convert|$)", text, re.DOTALL)
        material = re.search(
            r"Material:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|Material|Convert|$)", text, re.DOTALL)
        convert = re.search(
            r"Convert:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|Material|Convert|$)", text, re.DOTALL)
        input_ = re.search(
            r"Input:\s*(?:```|`)?(.+?)(?:```|`)?\s*(Observation|Final|Input|Thought|Question|Material|Convert|$)", text, re.DOTALL)
        final_thought = re.search(
            r"Final Thought:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|Material|Convert|$)", text, re.DOTALL)
        final_answer = re.search(
            r"Final Answer:\s*(.+?)\s*(Observation|Final|Input|Thought|Question|Material|Convert|$)", text, re.DOTALL)
        
        if (not input_) and (not final_answer):
            raise ValueError(f'unknown format from LLM: {text}')
        
        return {
            'Thought': (thought.group(1) if thought else None),
            'Material': (material.group(1) if material else None),
            'Convert': (convert.group(1) if convert else None),
            'Input': (input_.group(1).strip() if input_ else None),
            'Final Thought' : (final_thought.group(1) if final_thought else None),
            'Final Answer': (final_answer.group(1) if final_answer else None),
        }
    
    def _clear_name(self, text:str) -> str:
        remove_list = ['_clean_h', '_clean', '_charged', '_manual', '_ion_b', '_auto', '_SL', ]
        str_remove_list = r"|".join(remove_list)
        return re.sub(rf"({str_remove_list})", "", text)
    
    # @staticmethod
    def _get_atoms(self, material: str):
        ls_st: str[Path] = search_file(f'{material}*.cif', self.data_dir)
        if not ls_st:
            raise ValueError(f'{material} does not exists.')

        path: Path = ls_st[0]
        atoms = ase.io.read(path)
        return atoms
    
    def _write_log(self, action, text, run_manager):
        run_manager.on_text(f"\n[ASE repl] {action}: ", verbose=self.verbose)
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
        max_iteration = config['max_iteration']

        input_ = self._clear_name(inputs[self.input_key])

        for i in range(max_iteration + 1):
            llm_output = self.llm_chain.run(
                question=input_,
                agent_scratchpad = agent_scratchpad,
                callbacks=callbacks,
                stop=['Observation:', 'Question:',]
            )

            if not llm_output.strip():
                agent_scratchpad += 'Thought: '
                llm_output = self.llm_chain.run(
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
                    question=input_,
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
                check_sentence = ''
                if re.search(r'nothing', final_answer):
                    # please use tool `predictor` to get answer.'
                    final_answer = 'There are no data in ASE repl.'
                elif final_answer.endswith('.'):    
                    final_answer += check_sentence
                else:
                    final_answer = f'The answer for question "{input_}" is {final_answer}.{check_sentence}'
                return {self.output_key: final_answer}
            
            elif i >= max_iteration:
                final_answer = 'There are no data in ASE repl'
                self._write_log('Final Thought', final_answer, run_manager)
                return {self.output_key: final_answer}

            else:
                self._write_log('Material', output['Material'], run_manager)
                self._write_log('Convert', output['Convert'], run_manager)
                self._write_log('Thought', output['Thought'], run_manager)
                self._write_log('Input', output['Input'], run_manager)
                
            if output['Material']:
                atoms = self._get_atoms(output['Material'])

            pytool = PythonAstREPLTool(locals={'atoms': atoms})
            observation = str(pytool.run(output['Input'])).strip()
            
            if return_observation:
                if "\n" in observation:
                    self._write_log('Observation', "\n"+observation, run_manager)
                else:
                    self._write_log('Observation', observation, run_manager)
                return {self.output_key: observation}

            if "\n" in observation:
                self._write_log('Observation', "\n"+observation, run_manager)
            else:
                self._write_log('Observation', observation, run_manager)

            if output['Material'] and output['Convert']:
                agent_scratchpad += 'Material: {}\nConvert: {}\n'\
                    .format(output['Material'], output['Convert'])

            agent_scratchpad += 'Thought: {}\nInput: {} \nObservation: {}\n'\
                .format(output['Thought'], output['Input'], observation)

        raise AssertionError('Code Error! please report to author!')

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: str = ASE_PROMPT,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['question', 'agent_scratchpad'],
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        return cls(llm_chain=llm_chain, **kwargs)
