from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from chatmof.agents.agent import ChatMOF
from langchain.callbacks import StdOutCallbackHandler


if __name__ == '__main__':
    question = "What is the surface area of ACOGEF_clean?"
    #question = "Tell me structures that has void fraction between 0.2 and 0.3"
    #question = "What is top-3 structures that have high surface area?"
    #question = "Tell me top-5 structures that has void fraction near 0.7."
    #question = "Does ACOGEF_clean has open metal site or not?"
    #question = "What is the hydrogen uptake of ACOGEF_clean?"

    verbose = True
    search_internet = False

    llm = ChatOpenAI(temperature=0)
    callback_manager = [StdOutCallbackHandler()]

    chatmof = ChatMOF.from_llm(
        llm=llm, 
        verbose=verbose, 
    )
    output = chatmof.run(question, callbacks=callback_manager)