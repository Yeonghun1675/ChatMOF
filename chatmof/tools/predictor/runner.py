import os
from pathlib import Path
import re
import json
from typing import List
from pathlib import Path
from pydantic import BaseModel
from chatmof import __root_dir__
from chatmof.config import config
from chatmof.tools.predictor.utils import predict, model_names, search_file


class MOFTransformerRunner(BaseModel):
    model_dir: str = config['model_dir']
    data_dir: str = config['data_dir']
    verbose: bool = False

    def run(self, 
            prop: str, 
            material: str):
        if prop not in model_names:
            raise ValueError(f'property should be one {model_names}, not {property}')

        #params = os.path.join(self.model_dir, f'{prop}/hparams.yaml')
        model_dir = Path(self.model_dir)/prop
        data_list = self.parse_data(material)

        output = predict(
            data_list=data_list,
            model_dir=model_dir,
            verbose=self.verbose,
        )

        with (model_dir/'model_info.json').open() as f:
            model_info = json.load(f)

        if 'regression_logits' in output:
            cif_id = output['cif_id']
            logits = output['regression_logits']
        elif 'classification_logits_index' in output:
            label_json = model_dir/'label.json'
            if not label_json.exists():
                raise FileNotFoundError(f'There are no "label.json" in {model_dir}')
            with label_json.open() as f:
                labels = json.load(f)
            
            cif_id = output['cif_id']
            logits = [labels[i] for i in output['classification_logits_index']]
            
        return cif_id, logits, model_info
        
    def parse_data(self, material:str) -> List[Path]:
        data_dir = Path(self.data_dir)
        data_list = []

        s_mat = re.split(r",\s*", material)
        for mat in s_mat:
            if not mat.endswith('.cif'):
                if mat.endswith('*'):
                    mat = f'{mat}.cif'
                else:
                    mat = f'{mat}*.cif'

            if f_mat := search_file(mat, data_dir):
                data_list.extend(f_mat)

        if len(data_list) < len(s_mat):
            raise ValueError(f'There are no data in {material}')
        
        return data_list
    

if __name__ == '__main__':
    from chatmof.config import config

    runner = MOFTransformerRunner(
        model_dir=config['model_dir'], 
        data_dir=config['data_dir']
    )
    output = runner.run('solvent_removal_stability', 'PUG*')
    print (output)