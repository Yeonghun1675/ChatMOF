import re
import copy
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Iterable
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.agents import create_pandas_dataframe_agent

from chatmof import __root_dir__
from chatmof.config import config
from chatmof.tools.predictor.runner import MOFTransformerRunner
from chatmof.tools.predictor.utils import _predictable_properties, model_names
from chatmof.tools.genetic_algorithm.genetic_algorithm import GeneticAlgorithmChain
from chatmof.tools.genetic_algorithm.prompt import PLAN_PROMPT
from chatmof.tools.genetic_algorithm.cif_generate import CIFGenerator


class Generator(Chain):
    llm: BaseLanguageModel
    llm_chain: LLMChain
    generator_chain: GeneticAlgorithmChain
    topologies: List[str] = ['pcu']
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
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

        llm_output = self.llm_chain.predict(
           question=inputs[self.input_key],
           model_names=model_names,
           stop=['Question:'],
           callbacks=callbacks
        )

        output = self._parse_data(llm_output)
        
        run_manager.on_text(f'\nPlan for Generator : ', color='green')
        for key, answer in output.items():
            run_manager.on_text(f"\n{key}: ", verbose=self.verbose)
            run_manager.on_text(answer, verbose=self.verbose, color='yellow')

        run_manager.on_text('\n\nCycle 1.\n', color='green')

        for topology in self.topologies:
            run_manager.on_text('\nPredict property:\n', verbose=self.verbose, color='green')
            df = self.run_predictor(prop_text=output['Property'], topology=topology, data_dir=config['hmof_dir'])
            agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df = df,
                verbose=self.verbose,
                callback_manager=callbacks,
                return_intermediate_steps=True,
                max_iterations=1,                
            )            
            
            run_manager.on_text('\nSearch:', verbose=self.verbose, color='green')
            prompt = output['Search'] +\
                ' You must print in the form of a list of cif_id and properties with header. '\
                'For example) cif_id, bandgap\nacs+N12+E1, 0.21\nacs+N42+E42, 0.644'
            search_output = agent(prompt)

            parents = self._parse_predictor(search_output)
            
            run_manager.on_text('\nGenerate: \n', verbose=self.verbose, color='green')
            new_output = self.generator_chain.run(
                question=output['Generate'],
                parents=parents,
            )

            run_manager.on_text('\nBuild CIF: \n', verbose=self.verbose, color='green')
            generator = CIFGenerator(config['generate_dir'])
            generator.run(topology=topology, cif_list=new_output)


        df_gen = self.run_predictor(
            prop_text=output['Property'], 
            topology=topology, 
            data_dir=config['generate_dir']
        )

        df_final = copy.deepcopy(df)
        df_final.merge(df_gen, on='cif_id', how='outer')
        print (df_gen)

        final_agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df = df_final,
            verbose=self.verbose,
            callback_manager=callbacks,
        )

        final_output = final_agent.run(output['Final Thought'] + ' Use print when using python_repl_ast.')
        return {self.output_key: final_output,
                'origin_df': df,
                'generate_df': df_gen,
                }    

    def _parse_data(self, text):
        thought = re.search(r"Thought:\s*(.+?)\s*\n", text, re.DOTALL)
        search = re.search(r"Search look-up table:\s*(.+?)\s*\n", text, re.DOTALL)
        prop = re.search(r"Property:\s*(.+?)\s*\n", text, re.DOTALL)
        genetic_algorithm = re.search(r"Genetic algorithm:\s*(.+?)\s*(\n|$)", text, re.DOTALL)
        final_thought = re.search(r"Final Thought:\s*(.+?)\s*(\n|$)", text, re.DOTALL)

        if not thought:
            raise ValueError(f'unknown format from LLM: {text}')
        if not search:
            raise ValueError(f'unknown format from LLM: {text}')
        if not genetic_algorithm:
            raise ValueError(f'unknown format from LLM: {text}')
        if not prop:
            raise ValueError(f'unknown format from LLM: {text}')
        if not final_thought:
            raise ValueError(f'unknown format from LLM: {text}')
        
        return {
            'Thought': thought.group(1),
            'Property': prop.group(1),
            'Search': search.group(1),
            'Generate': genetic_algorithm.group(1),
            'Final Thought': final_thought.group(1),
        }

    def _parse_property(self, prop_text):
        prop_ls = re.split(',\s*', prop_text)
        for prop in prop_ls:
            if prop not in _predictable_properties:
                raise ValueError(f'property should be one of [{model_names}], not {prop}')
        return prop_ls
    
    def run_predictor(self, prop_text: str, topology: str, data_dir: str) -> List[str]:
        runner = MOFTransformerRunner(
            model_dir=config['model_dir'],
            data_dir=data_dir,
            verbose=self.verbose,
        )
        
        material = f'{topology}*.cif'
        df_ls = []
        for prop in self._parse_property(prop_text):
            cif_id, logits = runner.run(prop, material)
            df = pd.DataFrame({'cif_id': cif_id, prop: logits})
            df_ls.append(df)

        df_total = df_ls[0]
        for df in df_ls[1:]:
            df_total.merge(df, on='cif_id', how='outer')

        return df

    def _parse_predictor(self, agent_output: Dict[str, Any]) -> List[List[str]]:
        tool_action, output = agent_output['intermediate_steps'][-1]
        if tool_action.tool != 'python_repl_ast':
            raise ValueError(f'{tool_action.tool} is not python_repl_ast')
        
        try:
            data_ls = json.loads(output)
            if isinstance(data_ls, list):
                return data_ls    
            elif isinstance(data_ls, dict):
                if 'cif_id' in data_ls:
                    return zip(*data_ls.values())
                else:
                    return data_ls.items()
            else:
                raise ValueError()

        except json.JSONDecodeError:
            data_ls = output.split('\n')
            if re.search(r'cif_id', data_ls[0]):
                return [re.split(r',\s*|\s+', line.strip()) for line in data_ls[1:] if line.strip()]
            else:
                return [re.split(r',\s*|\s+', line.strip()) for line in data_ls if line.strip()]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PLAN_PROMPT,
        **kwargs: Any
    ) -> Chain:
        prompt = PromptTemplate(
            template=prompt, 
            partial_variables={'model_names': model_names},
            input_variables=['question']
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        genetic_chain = GeneticAlgorithmChain.from_llm(llm)

        return cls(
            llm=llm,
            llm_chain=llm_chain, 
            generator_chain=genetic_chain,
            **kwargs)


if __name__ == '__main__':
    from langchain.chat_models import ChatOpenAI
    from matplotlib import pyplot as plt
    
    llm = ChatOpenAI(temperature=0)
    #chain = GeneticAlgorithmChain.from_llm(llm, verbose=True)

    prompt = 'generate structures with bandgap near 0.5.'
    #prompt = 'generate structures with highest bandgap'
    #prompt = 'create a new material with a surface area close to 0.9 from 200 materials'
    chain = Generator.from_llm(llm, verbose=True)
    output = chain({'question':prompt})
    print (output['answer'])

    df = output['origin_df']
    df2 = output['generate_df']
    
    plt.hist(df['bandgap'], density=True, alpha=0.7, color='royalblue')
    plt.hist(df2['bandgap'], density=True, alpha=0.7, color='salmon')
    plt.legend(['original', 'generate'])
    plt.xlabel('Band-gap (eV)', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.show()


    