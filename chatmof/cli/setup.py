class CLICommand:
    """
    Inital setup for ChatMOF. Download coremof, hmof, and load_models
    """

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        from chatmof.setup_module import setup
        
        setup()