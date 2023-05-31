import io
import logging
from contextlib import redirect_stdout
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict
import yaml

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from moftransformer.modules import Module
from moftransformer.utils.validation import _IS_INTERACTIVE
from chatmof.tools.predictor.dataloader import load_datamodule


def read_yaml(config: str) -> Dict[str, Any]:
    path = Path(config)
    if not path.exists():
        raise FileNotFoundError('Config file does not existed in yaml file')
    elif path.suffix != '.yaml':
        raise TypeError(f'config file must be *.yaml, not {path.suffix}')
    
    with open(config) as f:
        file = yaml.load(f, Loader=yaml.Loader)

    return file['config']


def update_config(_config: Dict[str, Any]) -> None:
    pl.seed_everything(_config["seed"])
    _config['max_epochs'] = 1
    _config['devices'] = 1


def load_trainer(_config: Dict[str, Any]):
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


def _predict(
        dataloader: DataLoader, 
        model: Module,
        trainer: pl.Trainer,
) -> List[str]:

    rets = trainer.predict(model, dataloader)
    keys = rets[0].keys()

    output = defaultdict(list)
    for ret in rets:
        for key, value in ret.items():
            output[key].extend(value)

    return output


def predict(
        data_dir: str,
        data_list: str,
        params: str,
        verbose: bool = False
) -> Dict[str, List[str]]:
    
    if not verbose:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)    

    config = read_yaml(params)
    update_config(config)

    dataloader = load_datamodule(data_list, data_dir, config)
    model = Module(config)
    trainer = load_trainer(config)

    output = _predict(dataloader, model, trainer)
    return output
    

if __name__ == '__main__':
    params = '/home/dudgns1675/autogpt/ChatMOF/chatmof/database/load_model/bandgap/hparams.yaml'
    data_dir = '/home/dudgns1675/autogpt/ChatMOF/chatmof/database/structures/raw'
    data_list = 'ABAVIJ_clean\nZUZZEB_clean'

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    config = read_yaml(params)
    update_config(config)

    dataloader = load_datamodule(data_list, data_dir, config)
    model = Module(config)
    #print(buf.getvalue())

    trainer = load_trainer(config)

    ret = _predict(dataloader, model, trainer)
    #print(buf.getvalue())

    #output = buf.getvalue()
    #print (output)

