import os
import re
from pydantic import BaseModel
from chatmof import __root_dir__
from chatmof.tools.predictor.utils import predict


class Predictor(BaseModel):
    """Tools that predict properties using MOFTransformer.
    properties 
    """
    verbose: bool = False
    model_dir: str = os.path.join(__root_dir__, 'database/load_model')
    sep: str = ','

    def cleanup(self, query: str) -> str:
        query = re.sub(r"(^[\'\"])|([\'\"]$)", "", query)    # remove ' and "
        return query

    def run(
            self,
            query: str,
    ) -> str:
        """predict properties from cif-files"""
        query = self.cleanup(query)
        prop, *data_list = [
            q.strip() for q in query.split(self.sep)
        ]

        data_dir = os.path.join(__root_dir__, 'database/structures/raw')
        params = os.path.join(self.model_dir, f'{prop}/hparams.yaml')

        print ("\n")
        output = predict(
            data_dir=data_dir,
            data_list=data_list,
            params=params,
            verbose=self.verbose,
        )

        cif_id = output['cif_id']
        logits = output['regression_logits']
        
        string = ""
        for cif, logit in zip(cif_id, logits):
            string += f'{cif} : {logit} eV\n'
        string += 'Please check that these result lead to a final answer.'
        return string
    

if __name__ == '__main__':
    Predictor(verbose=True).run(data_list="ACOGEF_clean", property="bandgap")