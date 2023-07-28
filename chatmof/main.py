from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from chatmof.agents.agent import ChatMOF
from chatmof.config import config


if __name__ == '__main__':
    #question = "What is the surface area of ACOGEF?"
    #question = "Tell me structures that has void fraction between 0.2 and 0.3"
    #question = "What is top-3 structures that have high PLD?"
    #question = "Can you tell me the type of open metal in DUQYEU01?"
    #question = "Tell me top-5 structures that has void fraction near 0.7."
    #question = 'Generate the structures with highest surface area and visualize it.'
    #question = "Does UMODEH02_clean has open metal site or not?"
    #question = "What's the largest cavity diameter in materials with a density greater than a 7.0?"
    #question = "generate MOF with volume fraction == 0.5"
    #question = "What is the bandgap of ACOGEF_clean?"
    #question = "What is the hydrogen uptake of ACOGEF_clean?"
    #question = "How does the non-accessible surface area of QEZZEC compare with other materials?"
    #question = "What is the pore volume and surface area of ACOGEF_clean?"
    #question = "Can you predict all bandgap prediction in coremof?"
    #question = "What is the highest bandgap MOF?"
    #question = "What is the bandgap of ACOGEF_clean and ABETAE_clean?"
    #question = "What is the void fraction of ACOGEF_clean and ABETAE_clean?"
    #question = "Can you check that ACOGEF is in directory /home/dudgns1675/autogpt/ChatMOF/chatmof/database/structures/raw ?"
    #question = "Can you provide the top 10 materials with the highest largest free pore diameter?"
    #question = 'What is the metal type in TUYNEI?'
    question = "Can you generate the structure with accessible volume fraction with 0.5 and visualize it?"

    verbose = True
    search_internet = False

    llm = ChatOpenAI(temperature=config['temperature'], model='gpt-3.5-turbo')
    callback_manager = [StdOutCallbackHandler()]

    chatmof = ChatMOF.from_llm(
        llm=llm, 
        verbose=verbose, 
        search_internet=search_internet,
    )
    output = chatmof.run(question, callbacks=callback_manager)
