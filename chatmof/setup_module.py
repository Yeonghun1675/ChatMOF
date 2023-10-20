import subprocess
from pathlib import Path
import shutil
from moftransformer.utils.download import _download_file
from chatmof.config import config


class InstallationError(Exception):
    pass


def download_load_model(direc=None):
    if not direc:
        direc = Path(config['model_dir']).parent  # parent directory
    else:
        direc = Path(direc)

    if not direc.is_dir():
        raise ValueError(f"direc must be path for directory, not {direc}")
    if not direc.exists():
        direc.mkdir(parents=True, exist_ok=True)

    # load_model
    link = "https://figshare.com/ndownloader/files/42642739"
    name = 'load_model'
    filename = direc / f"{name}.zip"
    _download_file(link, filename, name)

    if not filename.exists():
        raise InstallationError(f'{name} does not downloaded.')
    
    shutil.unpack_archive(str(filename), format='zip')


def download_hmof(direc=None):
    if not direc:
        direc = Path(config['hmof_dir']).parent  # parent directory
    else:
        direc = Path(direc)

    if not direc.is_dir():
        raise ValueError(f"direc must be path for directory, not {direc}")
    if not direc.exists():
        direc.mkdir(parents=True, exist_ok=True)

    # load_model
    link = "https://figshare.com/ndownloader/files/42633475"
    name = 'hMOF'
    filename = direc / f"{name}.zip"
    _download_file(link, filename, name)

    if not filename.exists():
        raise InstallationError(f'{name} does not downloaded.')
    
    shutil.unpack_archive(str(filename), format='zip')


def download_coremof(direc=None):
    if not direc:
        direc = Path(config['data_dir']).parent  # parent directory
    else:
        direc = Path(direc)

    if not direc.is_dir():
        raise ValueError(f"direc must be path for directory, not {direc}")
    if not direc.exists():
        direc.mkdir(parents=True, exist_ok=True)

    # load_model
    link = "https://figshare.com/ndownloader/files/42808069"
    name = 'coremof'
    filename = direc / f"{name}.zip"
    _download_file(link, filename, name)

    if not filename.exists():
        raise InstallationError(f'{name} does not downloaded.')
    
    shutil.unpack_archive(str(filename), format='zip')


def setup():
    download_coremof()
    download_hmof()
    download_load_model()