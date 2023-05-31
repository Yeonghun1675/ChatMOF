import os
from pydantic import BaseModel
from functools import partial
from moftransformer.modules import Module
from chatmof import __root_dir__
from chatmof.tools.predictor.utils import predict


class Predictor(BaseModel):
    """Tools that predict properties using MOFTransformer.
    properties 
    """
    verbose: bool = False
    model_dir: str = os.path.join(__root_dir__, 'database/load_model')

    def run(
            self,
            data_list: str, 
            property: str,

    ) -> str:
        """predict properties from cif-files"""

        data_dir = os.path.join(self.model_dir, f'{property}/model.ckpt')
        params = os.path.join(self.model_dir, f'{property}/hparams.yaml')

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
            string += f'{cif} : {logit} m^2/g\n'

        return string