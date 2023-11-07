import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import argparse
import copy
from itertools import chain
import warnings

from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManagerForChainRun
from chatmof.config import config as config_default
from chatmof.agents.agent import ChatMOF
from chatmof.llm import get_llm

warnings.filterwarnings(action='ignore')


str_kwargs_names = {
    'accelerator': "Device name for MOFTransformer. accelerator must be one of [cuda, gpu, cpu] (default: cuda)",
    'logger': 'logger for generation task (Default: generate_mof.log)',
}

int_kwargs_names = {
    'max_length_in_predictor': 'max number of MOFs in predictor step. (default: 30)',
    'num_genetic_cycle': 'number of genetic algorithm cycle (default: 3)',
}

float_kwargs_names = {
}

bool_kwargs_names = {
    'search_internet': 'If True, using "search internet" tools. (default = False)',
    'verbose': 'If True, print intermediate step. (default = True)'
}


def main(model='gpt-4', temperature=0.1, **kwargs) -> str:
    config = copy.deepcopy(config_default)

    config.update(kwargs)
    search_internet = config['search_internet']
    verbose = config['verbose']

    llm = get_llm(model, temperature=temperature)
    callback_manager = [StdOutCallbackHandler()]

    run_manager = CallbackManagerForChainRun.get_noop_manager()

    chatmof = ChatMOF.from_llm(
        llm=llm, 
        verbose=verbose, 
        search_internet=search_internet,
    )

    print ('#' * 50 + "\n")
    print ('Welcom to ChatMOF!')
    print ("\n" + "#"*10 + ' Question ' + "#"*30)
    print ('Please enter the question below >>')
    question = input()
    
    output = chatmof.run(question, callbacks=callback_manager)

    print ('\n')
    print ("#"*10 + ' Output ' + "#" * 30)
    print (output)
    print ('\n')
    print ('Thanks for using CHATMOF!')

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add(
        '--model',
        '--model-name',
        '-m',
        type=str,
        help="OpenAI model name. model_name must be one of [gpt-4, gpt-3.5-turbo, gpt-3.5-turbo-16k]. (default: gpt-4)",
        default='gpt-4'
    )
    
    add(
        '--temperature',
        '-t',
        help='Temperature of LLM models. The lower the temperature, the more accurate the answer, and the higher the temperature, the more variable the answer. (default: 0.1)',
        type=float,
        default=0.1,
    )

    for key, value in str_kwargs_names.items():
        parser.add_argument(f"--{key}", type=str, required=False, help=f"(optional) {value}")

    for key, value in int_kwargs_names.items():
        parser.add_argument(f"--{key}", type=int, required=False, help=f"(optional) {value}")

    for key, value in float_kwargs_names.items():
        parser.add_argument(f"--{key}", type=float, required=False, help=f"(optional) {value}")

    for key, value in bool_kwargs_names.items():
        parser.add_argument(f"--{key}", action='store_true', required=False, help=f"(optional) {value}")

    args = parser.parse_args()
    model = args.model
    temperature = args.temperature

    kwargs = {}
    for key in chain(str_kwargs_names.keys(), 
                        int_kwargs_names.keys(),
                        float_kwargs_names.keys(),
                        bool_kwargs_names.keys(),
                        ):
        if value := getattr(args, key):
            kwargs[key] = value

    main(model=model, temperature=temperature, **kwargs)
