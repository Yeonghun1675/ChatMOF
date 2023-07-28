import os
import re
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional

import ase.io
import ase.visualize
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun

from chatmof.config import config
from chatmof.utils import search_file
from chatmof.tools.visualizer.prompt import PROMPT


class Visualizer(Chain):
    """Tools that search csv using Pandas agent"""
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
    
    def _write_log(self, action, text, run_manager):
        run_manager.on_text(f"\n[Visualizer] {action}: ", verbose=self.verbose)
        run_manager.on_text(text, verbose=self.verbose, color='yellow')

    def _parse_output(self, text:str) -> Dict[str, Any]:
        thought = re.search(r"(?<!Final )Thought:\s*(.+?)\s*(Material|Thought|Question|$)", text, re.DOTALL)
        material = re.search(r"Material:\s*(?:```|`)?(.+?)(?:```|`)?\s*(Material|Thought|Question|$)", text, re.DOTALL)

        if not material:
            raise ValueError(f'unknown format for LLM: {text}')
        
        return{
            'Thought': (thought.group(1) if thought else None),
            'Material': (material.group(1) if material else None),
        }

    def _visualize(self, f_st: Path) -> None:
        atoms = ase.io.read(f_st)
        ase.visualize.view(atoms)
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        
        llm_output = self.llm_chain.run(
            question = inputs[self.input_key],
            callbacks=callbacks,
            stop=['Question:'],
        )

        output = self._parse_output(llm_output)
        self._write_log('Thought', output['Thought'], run_manager)
        self._write_log('Material', output['Material'], run_manager)

        materials = output['Material'].split(',')
        for material in materials:
            material = material.strip()
            ls_st = search_file(f'**/**/{material}.cif', self.data_dir)
            if not ls_st:
                return {self.output_key: f'There are no structures with {material}.'}
            
            if '*' in material:
                for f_st in ls_st:
                    self._visualize(f_st)
            else:
                f_st = ls_st[0]
                self._visualize(f_st)

        return {self.output_key: f'The visualizer has successfully visualized the structure {materials}.'}
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: str = PROMPT,
        **kwargs
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['question'],
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        return cls(llm_chain=llm_chain, **kwargs)