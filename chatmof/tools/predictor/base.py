import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import OutputParserException

from chatmof import __root_dir__
from chatmof.config import config
from chatmof.tools.predictor.utils import model_names
from chatmof.tools.predictor.runner import MOFTransformerRunner


class Predictor(Chain):
    """Tools that predict properties using MOFTransformer.
    """
    llm_chain: LLMChain
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
    
    def parse_output(self, llm_output):
        output = llm_output
        return output
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            callbacks=callbacks
        )

        output = self.parse_output(llm_output)

        for prop, material in output:
            runner = MOFTransformerRunner(
                model_dir=self.model_dir, 
                data_dir=self.data_dir,
                verbose=self.verbose,
            )

        runner.run(llm_output)
    


if __name__ == '__main__':
    Predictor(verbose=True).run(data_list="ACOGEF_clean", property="bandgap")