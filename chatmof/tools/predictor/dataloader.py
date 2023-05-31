from functools import partial
from torch.utils.data import DataLoader
from chatmof.tools.predictor.dataset import ChatDataset


def load_datamodule(data_list, data_dir, _config):
    dataset = ChatDataset(
        data_list=data_list,
        data_dir=data_dir,
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