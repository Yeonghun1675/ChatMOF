import os
from typing import Dict, List
from pathlib import Path
import torch
from moftransformer.datamodules import Dataset as MOFDataset

from chatmof import __root_dir__


class ChatDataset(MOFDataset):
    def __init__(
            self,
            data_list: List[str],
            data_dir: str = os.path.join(__root_dir__, 'database/structures/raw'),
            nbr_fea_len: int = 64,
    ) -> Dict[str, torch.Tensor]:
        
        super(MOFDataset, self).__init__()  # inheritance TorchDatset
        self.data_dir = data_dir
        self.draw_false_grid = False
        self.split = ''
        self.nbr_fea_len = nbr_fea_len
        self.tasks = {}

        if data_list == 'all':
            self.cif_dis = [
                cif.stem 
                for cif in Path(data_dir).glob('*.cif')
                ]
        else:
            self.cif_ids = [
                data.replace(".cif", "")
                for data in data_list
                ]
        
        self.targets = [1] * len(self.cif_ids)
