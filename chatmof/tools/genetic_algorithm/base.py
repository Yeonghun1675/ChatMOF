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
    topologies: List[str] = config['topologies']
    input_key: str = 'question'
    output_key: str = 'answer'

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _write_log(self, action, text, run_manager):
        run_manager.on_text(f"\n[Generator] {action}: ", verbose=self.verbose)
        run_manager.on_text(text, verbose=self.verbose, color='yellow')
    
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
        
        self._write_log('Thought', output['Thought'], run_manager)
        self._write_log('Objective', output['Objective'], run_manager)
        
        df_dict = dict()
        prop = output['Property']
        self._write_log('Predict Properties', prop+"\n", run_manager)
        for topology in self.topologies:
            df, info_ls = self.run_predictor(prop_text=prop, topology=topology, data_dir=config['hmof_dir'])
            df_dict[topology] = df

        save_path = Path(config['generate_dir'])/f'{prop}-0.csv'
        pd.concat(df_dict.values()).to_csv(str(save_path))

        for cycle in range(1, config['num_genetic_cycle']+1):
            run_manager.on_text(f'\n\n[Generator] Run genetic algorithm : Cycle {cycle}.\n', color='green')
            df_dict = self.run_genetic(df_dict, output, run_manager, cycle, info_ls)

        self._write_log('Final Thought', output['Final Thought'], run_manager)
        
        df_final = df_dict[self.topologies[0]]
        for topo in self.topologies[1:]:
            df_gen = df_dict[topo]
            df_final = df_final.merge(df_gen, how='outer')

        information = f'Information of models : {info_ls}. If unit or condition are existed, you must include it in the final output.'
        searcher = TableSearcher.from_dataframe(
            llm = self.llm,
            dataframe = df_final,
            verbose=self.verbose,
            run_manager=run_manager,
        )

        final_output = searcher.run(
            question=output['Final Thought'],
            run_manager=run_manager,
            information=information,
        )
        return {self.output_key: final_output}


    def run_genetic(self, df_dict, output, run_manager, cycle, info_ls):
        self._write_log('Find Parents', output['Search'], run_manager)

        prop = output['Property']
        direc = Path(config['generate_dir']) / f'{prop}-{cycle}'

        parent_dict = dict()
        information = f'Information of models : {info_ls}. If unit or condition are existed, you must include it in the final output.'

        for topology in self.topologies:
            searcher = TableSearcher.from_dataframe(
                llm=self.llm,
                dataframe = df_dict[topology],
                verbose=self.verbose,
                run_manager=run_manager,
            )            
            prompt = "{} (Objective: {}). If you need to set a range of data, you need to make it wide to make the data exist."\
                    .format(output['Search'], output['Objective'])
            search_output = searcher.run(
                question=prompt, 
                return_observation=True,
                run_manager = run_manager,
                information=information,
            )
            parents = self._parse_predictor(search_output)

            if not parents:
                continue
                
            parent_dict[topology] = parents

        if not parent_dict:
            raise ValueError('There are no parents') 
    
        self._write_log('Get Children', output['Generate'], run_manager)

        child_dict = dict()
        for topology in self.topologies:
            if not topology in parent_dict.keys():
                continue

            children = self.generator_chain.run(
                question=output['Generate'],
                parents=parent_dict[topology],
                run_manager=run_manager,
            )
            child_dict[topology] = children
            print (children)

        self._write_log('Generate Structures', '', run_manager)

        generator = CIFGenerator(direc)
        for topology in self.topologies:
            if topology not in child_dict.keys():
                continue
            generator.run(topology=topology, cif_list=child_dict[topology])

        self._write_log('Predict Properties', output['Property']+"\n", run_manager)
        
        df_ls = []
        for topology in self.topologies:
            if topology not in child_dict.keys():
                continue
            try:
                df_gen, info_ls = self.run_predictor(
                    prop_text=output['Property'], 
                    topology=topology, 
                    data_dir=direc,
                )
            except ValueError:
                continue

            df_dict[topology] = df_dict[topology].merge(df_gen, how='outer')
            df_ls.append(df_gen)

        pd.concat(df_ls).to_csv(str(direc/f'../{prop}-{cycle}.csv'))
        return df_dict

    def _parse_output(self, text):
        thought = re.search(r"Thought:\s*(.+?)\s*\n", text, re.DOTALL)
        search = re.search(r"Search look-up table:\s*(.+?)\s*\n", text, re.DOTALL)
        prop = re.search(r"Property:\s*(.+?)\s*\n", text, re.DOTALL)
        objective = re.search(r"Objective:\s*(.+?)\s*\n", text, re.DOTALL)
        genetic_algorithm = re.search(r"Genetic algorithm:\s*(.+?)\s*(\n|$)", text, re.DOTALL)
        final_thought = re.search(r"Final Thought:\s*(.+?)\s*(\n|$)", text, re.DOTALL)

        if not thought:
            raise ValueError(f'unknown format from LLM: {text}')
        if not search:
            raise ValueError(f'unknown format from LLM: {text}')
        if not objective:
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
            'Objective': objective.group(1),
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
        info_ls = []
        for prop in self._parse_property(prop_text):
            cif_id, logits, model_info = runner.run(prop, material)
            df = pd.DataFrame({'cif_id': cif_id, prop: logits})
            df_ls.append(df)
            info_ls.append(model_info)

        df_total = df_ls[0]
        for df in df_ls[1:]:
            df_total.merge(df, on='cif_id', how='outer')

        return df, model_info

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
            pass

        label, parents = self._parse_markdown(output)
        return parents        
            
    def _parse_markdown(self, output: str) -> List[List[str]]:
        parents = []
        lines = [line.replace("|", "").strip() for line in output.split('\n')]
        label = lines[0].split()

        for line in lines[1:]:
            data = re.split(r',\s*|\s+', line)
            if len(data) < 2:
                continue
            elif "+" in data[0]:
                parents.append(data)
            elif "+" in data[1]:
                parents.append(data[1:])
            else:
                raise ValueError(output)
        return label, parents

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


    