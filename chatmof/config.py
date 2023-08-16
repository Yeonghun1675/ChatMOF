import os
from chatmof import __root_dir__


config = {
    # ChatMOF
    'search_internet': False,
    'verbose': True,

    # LLM - openAI
    'temperature': 0.1,    
    'model': 'gpt-4',

    # data directory
    'model_dir': os.path.join(__root_dir__, 'database/load_model/'),
    'structure_dir': os.path.join(__root_dir__, 'database/structures/'),
    'data_dir': os.path.join(__root_dir__, 'database/structures/coremof/'),
    'hmof_dir': os.path.join(__root_dir__, 'database/structures/hMOF/'),
    'generate_dir': os.path.join(__root_dir__, 'database/structures/generate'),

    # table searcher
    'lookup_dir': os.path.join(__root_dir__, 'database/tables/coremof.xlsx'),
    'max_iteration': 3,

    # building block searcher
    'buildingblock_dir' : os.path.join(__root_dir__, 'database/tables/mofkey.xlsx'),

    # predictor
    'max_length_in_predictor' : 30,
    'accelerator' : 'cuda',

    # generator
    'num_genetic_cycle': 3,
    'num_parents': 200,
    'logger': 'generate_mof.log',
    'topologies': ['pcu', 'dia', 'acs', 'rtl', 'cds', 'srs', 'ths', 'bcu', 'fsc'],
}