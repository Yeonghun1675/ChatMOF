from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from chatmof.agents.agent import ChatMOF
from chatmof.config import config


if __name__ == '__main__':
    #question = "What is the surface area of ACOGEF?"
    #question = "Tell me structures that has void fraction between 0.2 and 0.3"
    #question = "What is top-3 structures that have high PLD?"
    #question = "Can you recommand name of the sturcture with high surface area?"
    #question = "Tell me top-5 structures that has void fraction near 0.7."
    #question = "Does UMODEH02_clean has open metal site or not?"
    question = "generate MOF with highest surface area"
    #question = "What is the bandgap of ACOGEF_clean?"
    #question = "What is the hydrogen uptake of ACOGEF_clean?"
    #question = "What is the bandgap and surface area of ACOGEF_clean?"
    #question = "What is the pore volume and surface area of ACOGEF_clean?"
    #question = "Can you predict all bandgap prediction in coremof?"
    #question = "What is the highest bandgap MOF?"
    #question = "What is the bandgap of ACOGEF_clean and ABETAE_clean?"
    #question = "What is the void fraction of ACOGEF_clean and ABETAE_clean?"
    #question = "Can you check that ACOGEF is in directory /home/dudgns1675/autogpt/ChatMOF/chatmof/database/structures/raw ?"
    #question = "What's the Pore limiting diameter of GULWIU?"
    #question = 'What is the metal type in TUYNEI?'
    #question = 'What is the largest cavity diameter for EMUBOF?'

    verbose = True
    search_internet = False

    llm = ChatOpenAI(temperature=config['temperature'])
    callback_manager = [StdOutCallbackHandler()]

    chatmof = ChatMOF.from_llm(
        llm=llm, 
        verbose=verbose, 
    )
    output = chatmof.run(question, callbacks=callback_manager)
