import logging
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict
from functools import partial
import yaml

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from moftransformer.modules import Module
from moftransformer.utils.validation import _IS_INTERACTIVE

from chatmof.config import config as default_config
from chatmof.tools.predictor.dataset import ChatDataset


_predictable_properties = [
    path.stem for path in Path(default_config['model_dir']).iterdir() if not path.stem.startswith('__')
]

model_names = ",".join(_predictable_properties + ['else'])


def read_yaml(config: str) -> Dict[str, Any]:
    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(f'Config file does not existed in yaml file, {config}')
    elif path.suffix != '.yaml':
        raise TypeError(f'config file must be *.yaml, not {path.suffix}')
    
    with open(config) as f:
        file = yaml.load(f, Loader=yaml.Loader)

    return file['config']


def update_config(_config: Dict[str, Any], p_model: Path) -> None:
    pl.seed_everything(_config["seed"])
    _config['accelerator'] = default_config['accelerator']
    _config['load_path'] = str(p_model)
    _config['max_epochs'] = 1
    _config['devices'] = 1


def load_datamodule(
        data_list: List[Path], 
        _config: Dict[str, Any]
    ) -> DataLoader:

    dataset = ChatDataset(
        data_list=data_list,
        nbr_fea_len=_config['nbr_fea_len']
    )

    collate_fn = partial(
        ChatDataset.collate,
        img_size=_config['img_size']
    )

    return DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=_config['per_gpu_batchsize'],
        num_workers=_config['num_workers'],
    )


def load_trainer(_config: Dict[str, Any]) -> pl.Trainer:
    if _IS_INTERACTIVE:
        strategy = None
    elif pl.__version__ >= '2.0.0':
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=_config["devices"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy=strategy,
        max_epochs=_config["max_epochs"],
        log_every_n_steps=0,
        logger=False,
    )
    return trainer


def predict(
        data_list: List[Path],
        model_dir: Path,
        verbose: bool = False,
) -> Dict[str, List[str]]:
    
    #if not verbose:
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)    

    params = model_dir/'hparams.yaml'
    p_model, = model_dir.glob('*.ckpt')
    config = read_yaml(params)
    update_config(config, p_model)

    dataloader = load_datamodule(data_list, config)
    model = Module(config)
    trainer = load_trainer(config)

    rets = trainer.predict(model, dataloader)
    output = defaultdict(list)
    for ret in rets:
        for key, value in ret.items():
            output[key].extend(value)

    return output


def search_file(name: str, direc: Path) -> List[Path]:
    name = name.strip()
    if '*' in name:
        return list(direc.glob(name))
    
    f_name = direc/name
    if f_name.exists():
        return [f_name]
    
    return False