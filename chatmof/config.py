import os
from chatmof import __root_dir__


config = {
    # data directory
    'data_dir': os.path.join(__root_dir__, 'database/structures/coremof/'),
    'model_dir': os.path.join(__root_dir__, 'database/load_model/'),
    'lookup_dir': os.path.join(__root_dir__, 'database/tables/coremof.csv'),
    'hmof_dir': os.path.join(__root_dir__, 'database/structures/hMOF/'),
    'generate_dir': os.path.join(__root_dir__, 'database/structures/generate'),

    # LLM - openAI
    'temperature': 0,

    # MOFTransformer


    # genetic algorithm
    'logger': 'generate_mof.log',
    
}