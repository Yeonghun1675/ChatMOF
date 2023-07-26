import os
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import logging
from timeout_decorator import timeout

import pormake as pm
from moftransformer.utils.prepare_data import make_prepared_data
from chatmof.config import config


pm.log.disable_print()
pm.log.disable_file_print()   

DATABASE = pm.Database()
builder = pm.Builder()


def get_logger(filename):
    if os.path.exists(filename):
        os.unlink(filename)

    logger = logging.getLogger(filename)
    logger.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = get_logger(config['logger'])


class CIFGenerator(object):
    def __init__(self, save_dir:str) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
    def run(
        self, 
        topology: str,
        cif_list: Optional[List[str]] = None,
    ) -> None:
        topo = DATABASE.get_topo(topology)

        for cif in tqdm(cif_list, desc=topology):
            save_path = self.save_dir/f'{topology}+{cif}.cif'
            success = self._run_cif(cif, topo, save_path)                      

    def _run_cif(
            self,
            cif: str,
            topo: pm.Topology,
            save_path: Path,
    ):
        try:
            if not save_path.exists():
                current_cif = self._generate_cif(cif, topo)
                current_cif.write_cif(str(save_path))

            is_success = make_prepared_data(
                save_path, 
                self.save_dir, 
                logger=logger, 
                eg_logger=logger
            )

            if not is_success:
                save_path.unlink()
                return False
            else:
                return True
            
        except Exception as e:
            if save_path.exists():
                save_path.unlink()
            logger.error(e)
            return False

    @timeout(30) # timeout set to 30 second
    def _generate_cif(
        self,
        cif: str,
        topo: pm.Topology,
    ):
        bb_name1, bb_name2 = cif.split('+')
        bb1 = DATABASE.get_bb(bb_name1)
        bb2 = DATABASE.get_bb(bb_name2)

        # No metal cluster or more than 2 metal cluster
        if (bb1.has_metal and bb2.has_metal) or (not bb1.has_metal and not bb2.has_metal):
            return None   
        
        current_nodes = {}

        if bb_name2.startswith('E'):
            current_nodes[0] = bb1
            current_edges = {
                tuple(et): bb2 for et in topo.unique_edge_types
            }
        else:
            current_nodes[0] = bb1
            current_nodes[1] = bb2
            current_edges = {}

        current_mof = builder.build_by_type(topo, current_nodes, current_edges)
        return current_mof




