from chatmof.llm.llama import get_llama_llm
from chatmof.llm.openai import get_openai_llm
from chatmof.llm.codellama import get_codellama_llm
from langchain.base_language import BaseLanguageModel


def get_llm(model_name: str, temperature: float = 0.1, max_tokens: int = 4096) -> BaseLanguageModel:
    m_name = model_name.lower()

    if m_name.startswith('gpt'):
        llm = get_openai_llm(model_name, temperature=temperature)
    elif m_name.startswith('llama'):
        llm = get_llama_llm(model_name, max_token=max_tokens, temperature=temperature)
    elif m_name.startswith('codellama'):
        llm = get_codellama_llm(model_name, max_token=max_tokens, temperature=temperature)
    else:
        raise ValueError(f'model_name should be one of llama and gpt, not {model_name}')
    
    return llm
