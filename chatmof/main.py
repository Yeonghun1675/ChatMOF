from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from chatmof.agents.agent import ChatMOF
from langchain.callbacks import StdOutCallbackHandler


if __name__ == '__main__':
    #question = "What is the surface area of ACOGEF_clean?"
    #question = "Tell me structures that has void fraction between 0.2 and 0.3"
    #question = "What is top-3 structures that have high PLD?"
    #question = "Can you recommand name of the sturcture with high surface area?"
    #question = "Tell me top-5 structures that has void fraction near 0.7."
    #question = "Does ACOGEF_clean has open metal site or not?"
    #question = "What is the bandgap of ACOGEF_clean?"
    question = "What is the bandgap and surface area of ACOGEF_clean?"
    #question = "What is the bandgap of ACOGEF_clean and ABETAE_clean?"
    #question = "What is the void fraction of ACOGEF_clean and ABETAE_clean?"
    #question = "Can you check that ACOGEF is in directory /home/dudgns1675/autogpt/ChatMOF/chatmof/database/structures/raw ?"

    verbose = True
    search_internet = False

    llm = ChatOpenAI(temperature=0)
    callback_manager = [StdOutCallbackHandler()]

    chatmof = ChatMOF.from_llm(
        llm=llm, 
        verbose=verbose, 
    )
    output = chatmof.run(question, callbacks=callback_manager)
