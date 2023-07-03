import os
from pathlib import Path
import re
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

        if 'regression_logits' in output:
            cif_id = output['cif_id']
            logits = output['regression_logits']
        else:
            raise NotImplementedError()

        return cif_id, logits
        
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
    output = runner.run('bandgap', 'PUG*')
    print (output)