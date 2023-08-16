from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManagerForChainRun
from chatmof.config import config
from chatmof.agents.agent import ChatMOF


def main(**kwargs) -> str:
    config.update(kwargs)

    model = config['model']
    search_internet = config['search_internet']
    verbose = config['verbose']

    llm = ChatOpenAI(temperature=config['temperature'], model=model)
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
    main()