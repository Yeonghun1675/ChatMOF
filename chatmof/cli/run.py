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

        add(
            '--accelerator',
            '-a',
            help='Device name for MOFTransformer. accelerator must be one of [cuda, gpu, cpu]',
            type=str,
            default='cuda'
        )
        add(
            '--max-length-in-predictor',
            type=int,
            default=30,
        )
        add(
            '--num-genetic-cycle',
            type=int,
            default=3,
        )
        add(
            '--logger',
            help='logger for generation task (Default: generate_mof.log)',
            type=str,
            default='generate_mof.log'
        )


    @staticmethod
    def run(args):
        from chatmof.main import main
        kwargs = {}
        print (dict(args))
        print (args)
        main(kwargs)