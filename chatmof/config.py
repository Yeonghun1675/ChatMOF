import os
from chatmof import __root_dir__


config = {
    # data directory
    'data_dir': os.path.join(__root_dir__, 'database/structures/coremof/'),
    'model_dir': os.path.join(__root_dir__, 'database/load_model/'),
    'lookup_dir': os.path.join(__root_dir__, 'database/tables/coremof.csv'),
    'h_mof_dir': os.path.join(__root_dir__, 'database/structures/hmof/'),

    # LLM - openAI
    'temperature': 0,

    # MOFTransformer
    
}