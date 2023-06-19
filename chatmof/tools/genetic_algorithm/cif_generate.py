from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import pormake as pm
from moftransformer.utils.prepare_data import make_prepared_data, get_logger


DATABASE = pm.database()
builder = pm.Builder()
logger = get_logger('generate_mof.log')


class CIFGenerator(object):
    def __init__(self, save_dir:str) -> None:
        self.save_dir = Path(save_dir)

    def run(
        self, 
        topology: str,
        cif_list: Optional[List[str]] = None,
    ) -> None:
        topo = DATABASE.get_topo(topology)

        for cif in tqdm(cif_list, desc=topology):
            save_path = self.save_dir/f'{topology}+{cif}.cif'
            self._run_cif(self, cif, topo, save_path)                      

    def _run_cif(
            self,
            cif: str,
            topo: pm.Topology,
            save_path: Path,
    ):
        # if not save_path.exists():    
        #     try:
        #         material = self._generate_cif(self, cif, topo)
        #         if material:
        #             material.write_cif(str(save_path))
        #         else:
        #             return False
        #     except Exception as e:
        #         return False
        
        try:
            is_success = make_prepared_data(save_path, self.save_dir)
            if not is_success:
                save_path.unlink()
                return False
            else:
                return True
        except Exception as e:
            return False

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




