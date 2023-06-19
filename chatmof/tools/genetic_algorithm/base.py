import re
import numpy as np
from typing import Dict, Any, List, Optional, Iterable
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun

from chatmof import __root_dir__
from chatmof.tools.genetic_algorithm.genetic_algorithm import GeneticAlgorithmChain
from chatmof.tools.genetic_algorithm.prompt import PLAN_PROMPT

if __name__ == '__main__':
    N = 100
    E = 200

    V_N = np.random.rand(N)
    V_E = np.random.rand(E)


    def get_parents(num: int, topo:str='acs'):
        i = 0

        output = []
        while len(output) < num:
            n = np.random.randint(N)
            e = np.random.randint(E)

            parent = f'N{n}+E{e}'
            value = V_N[n] * V_E[e]

            v = (parent, value)
            if v not in output:
                output.append(v)
        
        return output

    def get_values(gene):
        values = []
        for g in gene:
            n, e = g.split('+')
            n = int(n[1:])
            e = int(e[1:])
            value = V_N[n] * V_E[e]
            values.append(value)
        return values


class Generator(Chain):
    llm_chain: LLMChain
    generator_chain: GeneticAlgorithmChain
    search_csv_chain: Chain
    predictor_chain: Chain
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_key(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()


if __name__ == '__main__':
    from langchain.chat_models import ChatOpenAI
    from matplotlib import pyplot as plt
    
    llm = ChatOpenAI(temperature=0)
    chain = GeneticAlgorithmChain.from_llm(llm, verbose=True)

    parents = get_parents(200)
    prompt = 'generate structures with value near 0.5.'
    #prompt = 'generate structures with highest value'
    output = chain.run(question=prompt, parents=parents)

    print (output)
    _, v_p = zip(*parents)
    v_o = get_values(output)
    
    plt.hist(v_p, density=True, alpha=0.7, color='royalblue')
    plt.hist(v_o, density=True, alpha=0.7, color='salmon')
    plt.legend(['original', 'regenerate'])
    plt.xlabel('Value', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.show()


    