from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.base_language import BaseLanguageModel


CHAT_MODEL = [
    'gpt-4',
    'gpt-4-0613',
    'gpt-4-32k',
    'gpt-4-32k-0613',
    'gpt-4-0314',
    'gpt-4-32k-0314',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-instruct',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-16k-0613',
    'gpt-3.5-turbo-0301',
]


MODEL = [
    'text-davinci-003',
    'text-davinci-002',
]


def get_openai_llm(model_name: str, temperature: float=0.1) -> BaseLanguageModel:
    model_name = model_name.lower()
    if model_name in CHAT_MODEL:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    elif model_name in MODEL:
        llm = OpenAI(model_name=model_name, temperature=temperature)

    return llm
