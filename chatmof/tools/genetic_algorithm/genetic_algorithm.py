import re
from typing import Dict, Any, List, Optional, Iterable
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun

from chatmof.config import config
from chatmof.tools.genetic_algorithm.prompt import GENETIC_PROMPT


class GeneticAlgorithmChain(Chain):
    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = GENETIC_PROMPT
    input_key: str = 'question'  #: :meta private:
    output_key: str = 'answer'  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key, 'parents']
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _process_llm_result(
        self,
        llm_output
    ) -> Dict[str, str]:
        
        children = []
        for child in re.split(r"(\n|,\s?|\s)", llm_output):
            if child := self._evaluate_child(child):
                children.append(child)
        return {self.output_key: children}

    def _evaluate_child(self, child):
        child = child.strip()
        #if gene := re.search(r'[a-z]+\+(N|E)\d+\+(N|E)\d+', child):
        if gene := re.search(r'(N|E)\d+\+(N|E)\d+', child):
            return gene.group(0)
        else:
            return None

    def parse_parents(self, parents:Iterable[Iterable[str]]):
        n_parents = config['num_parents']
        names = []

        for i, parent in enumerate(parents):
            if i> n_parents:
                break
            if len(parent) < 2:
                continue
            elif isinstance(parent, list):
                name, *_ = parent
                name = name.strip()
            elif isinstance(parent, dict):
                name, *_ = parent.values()
            
            names.append(name)

        return ", ".join(names) + "\n"
    
    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key])

        parents = inputs['parents']

        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            parents=self.parse_parents(parents),
            callbacks=_run_manager.get_child(),
        )

        return self._process_llm_result(llm_output,)

    @property
    def _chain_type(self) -> str:
        return "genetic_algorithm_chain"
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = GENETIC_PROMPT,
        **kwargs: Any,
    ) -> Chain:
        prompt = PromptTemplate(
            input_variables=['question', 'parents'],
            template=prompt,
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)