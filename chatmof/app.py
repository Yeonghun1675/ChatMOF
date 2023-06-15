import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from chatmof.agents.agent import ChatMOF
from langchain.callbacks import StdOutCallbackHandler

verbose = True
search_internet = False

llm = ChatOpenAI(temperature=0)
callback_manager = [StdOutCallbackHandler()]

chatmof = ChatMOF.from_llm(
    llm=llm, 
    verbose=verbose, 
)

st.title('Welcome to the ChatMOF 🤖')
input = st.text_input('Question for Metal-organic Framework!')

if input:
    text = chatmof.run(input)
    st.text(text)
