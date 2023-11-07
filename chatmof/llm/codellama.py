import torch
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.base_language import BaseLanguageModel


CodeLLAMA_HF = [
    'CodeLlama-7b-Python-hf',
    'CodeLlama-13b-Python-hf',
    'CodeLlama-34b-Python-hf',
]


def get_codellama_llm(model_name: str, max_token: int =4096, temperature: float=0.1) -> BaseLanguageModel:
    if model_name not in CodeLLAMA_HF:
        raise ValueError(f'model_name should be one of {CodeLLAMA_HF}, not {model_name}')
    model_name = f'codellama/{model_name}'

    tokenizer = CodeLlamaTokenizer.from_pretrained(model_name,
                                            use_auth_token=True,
                                            )
    model = LlamaForCausalLM.from_pretrained(model_name,
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
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                )

    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temperature})
    return llm



