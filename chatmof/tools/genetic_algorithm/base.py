import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
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
from chatmof.tools.search_csv.base import TableSearcher
from chatmof.tools.genetic_algorithm.genetic_algorithm import GeneticAlgorithmChain
from chatmof.tools.genetic_algorithm.prompt import PLAN_PROMPT
from chatmof.tools.genetic_algorithm.cif_generate import CIFGenerator


class Generator(Chain):
    llm: BaseLanguageModel
    llm_chain: LLMChain
    generator_chain: GeneticAlgorithmChain
    topologies: List[str] = ['pcu', 'acs']
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

        llm_output = self.llm_chain.run(
           question=inputs[self.input_key],
           model_names=model_names,
           stop=['Question:'],
           callbacks=callbacks
        )

        output = self._parse_output(llm_output)
        
        run_manager.on_text('\n[Generator] Thought: ', verbose=self.verbose)
        run_manager.on_text(output['Thought'], verbose=self.verbose, color='yellow')

        df_dict = dict()
        run_manager.on_text('\n[Generator] Predict Properties: ', verbose=self.verbose)
        run_manager.on_text(output['Property'] + "\n", verbose=self.verbose, color='yellow')
        for topology in self.topologies:
            df = self.run_predictor(prop_text=output['Property'], topology=topology, data_dir=config['hmof_dir'])
            df_dict[topology] = df

        for cycle in range(1, config['num_genetic_cycle']+1):
            run_manager.on_text(f'\n\n[Generator] Run genetic algorithm : Cycle {cycle}.\n', color='green')
            df_dict = self.run_genetic(df_dict, output, run_manager, cycle)

        run_manager.on_text('\n[Generator] Final Thought: ', verbose=self.verbose)
        run_manager.on_text(output['Final Thought'], verbose=self.verbose, color='yellow')
        
        df_final = df_dict[self.topologies[0]]
        for topo in self.topologies[1:]:
            df_gen = df_dict[topo]
            df_final.merge(df_gen, on='cif_id', how='outer')

        searcher = TableSearcher.from_dataframe(
            llm = self.llm,
            dataframe = df_final,
            verbose=self.verbose,
            run_manager=run_manager,
        )

        final_output = searcher.run(output['Final Thought'])
        return {self.output_key: final_output}


    def run_genetic(self, df_dict, output, run_manager, cycle):
        run_manager.on_text('\n[Generator] Find Parents: ', verbose=self.verbose)
        run_manager.on_text(output['Search'], verbose=self.verbose, color='yellow')
        prop = output['Property']
        direc = Path(config['generate_dir']) / f'{prop}-{cycle}'

        parent_dict = dict()
        for topology in self.topologies:
            searcher = TableSearcher.from_dataframe(
                llm=self.llm,
                dataframe = df_dict[topology],
                verbose=self.verbose,
            )            
            prompt = output['Search'] +\
                ' You must print line-by-line list of cif_id and properties line by line with header, not print `df` directly. '\
                'For example, output should like ```cif_id, bandgap\nacs+N12+E1, 0.21\nacs+N42+E42, 0.644```'
            search_output = searcher.run(
                question=prompt, 
                return_observation=True,
                run_manager = run_manager,
            )
            parents = self._parse_predictor(search_output)
            parent_dict[topology] = parents
            
        run_manager.on_text('\n[Generator] Get Children: ', verbose=self.verbose)
        run_manager.on_text(output['Generate'], verbose=self.verbose, color='yellow')

        child_dict = dict()
        for topology in self.topologies:
            children = self.generator_chain.run(
                question=output['Generate'],
                parents=parent_dict[topology],
                run_manager=run_manager,
            )
            child_dict[topology] = children

        run_manager.on_text('\n[Generator] Generate Structures: ', verbose=self.verbose)

        generator = CIFGenerator(direc)
        for topology in self.topologies:
            generator.run(topology=topology, cif_list=child_dict[topology])

        run_manager.on_text('\n[Generator] Predict Properties: ', verbose=self.verbose)
        run_manager.on_text(output['Property']+"\n", verbose=self.verbose, color='yellow')
        
        for topology in self.topologies:
            df_gen = self.run_predictor(
                prop_text=output['Property'], 
                topology=topology, 
                data_dir=direc,
            )
            df_dict[topology].merge(df_gen, on='cif_id', how='outer')
            df_gen.to_csv(str(direc/'output.csv'))
        
        return df_dict

    def _parse_output(self, text):
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
            data_dir=str(data_dir),
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

    def _parse_predictor(self, output: str) -> List[List[str]]: 
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
                return [re.split(r',\s*|\s+', line.replace("|", "").strip()) for line in data_ls[1:] if line.strip()]
            else:
                return [re.split(r',\s*|\s+', line.replace("|", "").strip()) for line in data_ls if line.strip()]

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


    