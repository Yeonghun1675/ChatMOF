import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain.base_language import BaseLanguageModel


LLAMA_HF = [
    'Llama-2-7b'
    'Llama-2-7b-hf'
    'Llama-2-7b-chat',
    'Llama-2-7b-chat-hf',
    'Llama-2-13b',
    'Llama-2-13b-hf',
    'Llama-2-13b-chat',
    'Llama-2-13b-chat-hf',
    'Llama-2-70b',
    'Llama-2-70b-hf',
    'Llama-2-70b-chat',
    'Llama-2-70b-chat-hf',
]


def get_llama_llm(model_name: str, max_token: int =4096, temperature: float=0.1) -> BaseLanguageModel:
    model_name = model_name.capitalize()
    if model_name not in LLAMA_HF:
        raise ValueError(f'model_name should be one of {LLAMA_HF}, not {model_name}')
    model_name = f'meta-llama/{model_name}'

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_auth_token=True,
                                            )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                use_auth_token=True,
                                                )

    pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                do_sample=True,
                top_k=30,
                max_new_tokens=max_token,
                eos_token_id=tokenizer.eos_token_id
                )

    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temperature})
    return llm



