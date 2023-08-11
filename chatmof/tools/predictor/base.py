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

from chatmof import __root_dir__
from chatmof.config import config
from chatmof.tools.search_csv.base import TableSearcher
from chatmof.tools.predictor.utils import model_names, _predictable_properties
from chatmof.tools.predictor.runner import MOFTransformerRunner
from chatmof.tools.predictor.prompt import (
    PROMPT, FINAL_MARKDOWN_PROPMT
)


class Predictor(Chain):
    """Tools that predict properties using MOFTransformer.
    """
    llm: BaseLanguageModel
    llm_chain: LLMChain
    final_single_chain: LLMChain
    model_dir: str= config['model_dir']
    data_dir: str = config['data_dir']
    tool_names: str = model_names
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _parse_output(self, text) -> Dict[str, Any]:
        thought = re.search(r"Thought:\s*(.+?)\s*\n", text, re.DOTALL)
        properties = re.findall(r"Property:\s*(.+?)\s*\n", text, re.DOTALL)
        materials = re.findall(r"Material:\s*(.+?)\s*\n", text, re.DOTALL)
        final_thought = re.search(r"Final Thought:\s*(.+?)\s*(\n|$)", text, re.DOTALL)

        if not thought:
            raise ValueError(f'unknown format from LLM: {text}')
        if not properties:
            raise ValueError(f'unknown format from LLM: {text}')
        if not materials:
            raise ValueError(f'unknown format from LLM: {text}')
        if not final_thought:
            raise ValueError(f'unknown format from LLM: {text}')
        if len(properties) != len(materials):
            raise ValueError('number of properties and materials are not the same: {text}')
        for prop in properties:
            if prop not in _predictable_properties:
                raise ValueError(f'Predictor can not predict the property: {prop}')
        
        return {
            'Thought': thought.group(1),
            'Property': properties,
            'Materials': materials,
            'Final Thought': final_thought.group(1),
        }            
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ):
        run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = run_manager.get_child()

        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            callbacks=callbacks
        )

        output = self._parse_output(llm_output)
        
        run_manager.on_text(f"\n[Predictor] Thought: ", verbose=self.verbose)
        run_manager.on_text(output['Thought'], verbose=self.verbose, color='yellow')

        df_ls = []
        info_ls = []
        for prop, mat in zip(output['Property'], output['Materials']):
            run_manager.on_text(f"\n[Predictor] Property: ", verbose=self.verbose)
            run_manager.on_text(prop, verbose=self.verbose, color='yellow')
            run_manager.on_text(f"\n[Predictor] Materials: ", verbose=self.verbose)
            run_manager.on_text(f"{mat}\n", verbose=self.verbose, color='yellow')

            runner = MOFTransformerRunner(
                model_dir=self.model_dir,
                data_dir=self.data_dir,
                verbose=self.verbose,
            )
            cif_id, logits, model_info = runner.run(prop, mat)
            df = pd.DataFrame({'cif_id': cif_id, prop: logits})
            df_ls.append(df)
            info_ls.append(model_info)
            
        df_total = df_ls[0]
        for df in df_ls[1:]:
            df_total.merge(df, on='cif_id', how='outer')

        information = f'Information of models : {info_ls}. If unit or condition are existed, you must include it in the final output.'

        run_manager.on_text(f"[Predictor] Final Thought: ", verbose=self.verbose)
        run_manager.on_text(output['Final Thought'], verbose=self.verbose, color='yellow')
        
        if len(df_total) < config['max_length_in_predictor']:
            final_output = self.final_single_chain.run(
                table = df_total.to_markdown(),
                information=information,
                question = output['Final Thought']
            )
        else:
            searcher = TableSearcher.from_dataframe(
                llm = self.llm,
                dataframe = df_total, 
                verbose = self.verbose
            )
            final_output = searcher.run(question=output['Final Thought'],
                                        information=information,
                                        run_manager=run_manager)
            
        return {self.output_key: final_output}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: str = PROMPT,
        final_single_prompt: str = FINAL_MARKDOWN_PROPMT,
        **kwargs: Any,
    ) -> Chain:
        template = PromptTemplate(
            template=prompt,
            input_variables=['question'],
            partial_variables={'model_names':model_names}
        )
        fs_template = PromptTemplate(
            template=final_single_prompt,
            input_variables=['table', 'information', 'question']
        )

        llm_chain = LLMChain(llm=llm, prompt=template)
        final_single_chain = LLMChain(llm=llm, prompt=fs_template)

        return cls(
            llm=llm,
            llm_chain=llm_chain, 
            final_single_chain=final_single_chain,
            **kwargs)


if __name__ == '__main__':
    from langchain.chat_models import ChatOpenAI
    from matplotlib import pyplot as plt
    
    llm = ChatOpenAI(temperature=0, model_name='gpt-4')

    prompt = 'Find the structure with the highest hydrogen diffusivity in a dilute system'

    tool = Predictor.from_llm(llm, verbose=True)
    output = tool.run(prompt)
    print (output)
