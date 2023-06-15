import numpy as np
from collections import defaultdict
import pormake as pm
import json
from tqdm import tqdm
from collections import Counter
import gc

pm.log.disable_print()
pm.log.disable_file_print()

database = pm.Database()

with open('topo_seed.json') as f:
    topo_seed = json.load(f)

num_mof_for_topo = 3000
builder = pm.Builder()

for topo_name, seed in topo_seed.items():
    #if topo_name in ['pcu', 'dia', 'nbo', 'pts', 'ths', 'srs', 'rtl', 'tbo',
    #                 'rna', 'lvt'
    #                 ]:
    #    continue
    if topo_name not in ['rna', 'lvt']:
        continue
    
    seed_1, seed_2 = seed.values()
    topo = database.get_topo(topo_name)

    ls_up = []
    num_mof = 0
    
    with tqdm(total=num_mof_for_topo, desc=topo_name) as pbar:
        while num_mof < num_mof_for_topo:
            bb_name1 = np.random.choice(seed_1)
            bb_name2 = np.random.choice(seed_2)
            
            bb1 = database.get_bb(bb_name1)
            bb2 = database.get_bb(bb_name2)

            if (bb1.has_metal or bb2.has_metal) and not (bb1.has_metal and bb2.has_metal):
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
                    #E0 = database.get_bb('E0')
                    #current_edges = {
                    #    tuple(et): E0 for et in topo.unique_edge_types
                    #}

                try:
                    current_mof = builder.build_by_type(topo, current_nodes, current_edges)
                    current_mof.write_cif(f'/home/dudgns1675/autogpt/ChatMOF/chatmof/database/structures/hMOF/{topo_name}+{bb_name1}+{bb_name2}.cif')
                    del current_mof
                    gc.collect()
                except:
                    pass

                num_mof += 1
                pbar.update(1)