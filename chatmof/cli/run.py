from itertools import chain


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


class CLICommand:
    """
    Run ChatMOF code

    ex) chatmof run model_name=gpt-3.5-turbo temperature=0.1 verbose=True
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument

        add(
            '--model-name',
            '-m',
            type=str,
            help="OpenAI model name. model_name must be one of [gpt-4, gpt-3.5-turbo, gpt-3.5-turbo-16k]. (default: gpt-4)",
            choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'],
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

    @staticmethod
    def run(args):
        from chatmof.main import main
        model = args.model_name
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