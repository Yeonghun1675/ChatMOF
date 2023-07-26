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


class Visualizer(Chain):
    """Tools that search csv using Pandas agent"""
    llm_chain: LLMChain
    df: pd.DataFrame
    encoder: tiktoken.core.Encoding
    num_max_data: int = 200
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _write_log(self, action, text, run_manager):
        run_manager.on_text(f"\n[Table Searcher] {action}: ", verbose=self.verbose)
        run_manager.on_text(text, verbose=self.verbose, color='yellow')
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        raise NotImplementedError()

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
            input_variables=['df_index', 'information', 'df_head', 'question', 'agent_scratchpad']
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
            input_variables=['df_index', 'information', 'df_head', 'question', 'agent_scratchpad']
        )
        llm_chain = LLMChain(llm=llm, prompt=template)
        encoder = tiktoken.encoding_for_model(llm.model_name)
        return cls(llm_chain=llm_chain, df=dataframe, encoder=encoder, **kwargs)