import os
from typing import Dict, List
from pathlib import Path
import torch
from moftransformer.datamodules import Dataset as MOFDataset

from chatmof import __root_dir__


class ChatDataset(MOFDataset):
    def __init__(
            self,
            data_list: List[Path],
            nbr_fea_len: int = 64,
    ) -> Dict[str, torch.Tensor]:
        
        super(MOFDataset, self).__init__()  # inheritance TorchDatset

        self.data_list = data_list
        self.draw_false_grid = False
        self.split = ''
        self.nbr_fea_len = nbr_fea_len
        self.tasks = {}

        self.cif_ids = [
            cif.stem for cif in data_list
        ]

        self.data_dir = data_list[0].parent
        
        self.targets = [0] * len(self.cif_ids)
