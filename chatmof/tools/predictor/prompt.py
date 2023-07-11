PROMPT = """plan to use machine learning to predict the properties of matter. To answer the question, you have to fill in the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Property: the property you can predict, should be one of [{model_names}]
Material: names of materials separated using comma. If you need to proceed for all material, write *. To proceed for a specific topology, append the topology name with an * (ex. pcu*)
... (this Property/Material can repeat N times)
Run Machine Learning: nothing to do
Final Thought: you should think about how you will derive a final answer from the results of machine learning.

Begin!

Question: predict the surface area and hydrogen uptake of ACOPEN and ELOBPE.
Thought: I need to gather data on ACOPEN and train a machine learning model to predict its surface area and hydrogen uptake.
Property: surface_area
Material: ACOPEN, ELOBPE
Property: hydrogen_uptake_at_100bar
Material: ACOPEN, ELOBPE
Final Thought: Based on the result, answer the question using predicted surface area and the predicted hydrogen uptake at 100 bar.

Question: which MOF has a highest band-gap?
Thought: I need to gather data on the band-gap of different structures and compare them to find the one with the highest value.
Property: bandgap
Material: *
Final Thought: Based on the result, find the structure with the highest predicted band-gap value.

Question: Predict surface area and save results in csv format, only pcu topology.
Thought: I need to gather data on the surface area of materials with pcu topology and train a machine learning model to predict their surface area. Then, I can use the model to make predictions and save the results in a csv format.
Property: surface_area
Material: pcu*
Final Thought: The predicted surface area values for materials with pcu topology can be found in the saved csv file.

Question: {question}"""


FINAL_MARKDOWN_PROPMT = """
You need to answer the question from the markdown table below

Markdown Table:
{table}
Question: {question}
Answer:"""


FINAL_DF_PROMPT = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should make a valid python command as input. You must use print the output using the `print` function at the end.

Use the following format:

Question: the input question you must answer
Input: the valid python code 
Observation: the result of python code

This is the result of `print(df.head())`:
{df_head}

Begin!

Question: We want to know the total number of columns in the dataframe.
Input: 
``` 
print (df.shape[0])
```
Observation: {df_shape}

Question: {question}
Input:
"""


FINAL_OUTPUT_PROMPT = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should make a valid python command as input. You must use print the output using the `print` function at the end.
You should use the `to_markdown` function when you print a pandas object.

Use the following format:

Question: the input question you must answer
Input: the valid python code 
Observation: the result of python code
Final answer: the final answer to the original input question

Begin!

Question: {question}
Input: {input}
Observation: {observation}
Final answer: """


FINAL_SEARCH_PROMPT = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should make a valid python command as input. You must use print the output using the `print` function at the end.
You should use the `to_markdown` function when you print a pandas object.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Input: the valid python code using the Pandas library
Observation: the result of python code
... (this Thought/Input/Observation can repeat N times)
Final Thought: you should think about how to answer the question based on your observation
Final Answer: the final answer (full sentence) to the original input question. If you can't answer the question, say `nothing`


This is the result of `print(df.head().to_markdown())`:
{df_head}

Begin!

Question: {question}
{agent_scratchpad}
"""



if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    import pandas as pd

    from chatmof.tools.predictor.utils import model_names

    llm = ChatOpenAI(temperature=0.0)
    #prompt = PromptTemplate(template=PROMPT, input_variables=['question', 'model_names'])
    #chain = LLMChain(llm=llm, prompt=prompt)
    #output = chain.run(
    #    question='Predict surface area and save results in csv format, only pcu topology.',
    #    model_names=model_names)
    #print (output)
    
    #prompt = PromptTemplate(template=FINAL_DF_PROMPT, input_variables=['df_head', 'df_shape', 'question'])
    #chain = LLMChain(llm=llm, prompt=prompt)
    #df = pd.DataFrame({'cif_id':['Material1'], 'surface area (m^2/g)':['10.234']}).set_index('cif_id')
    
    #output = chain.run(
    #    question = 'What is the surface area of Material1?',
    #    df_head = df.head(),
    ##    df_shape = df.shape[0],
    #    stop=['Observation:'],
    #)

    prompt = PromptTemplate(template=FINAL_OUTPUT_PROMPT, input_variables=['question', 'input', 'observation'])
    chain = LLMChain(llm=llm, prompt=prompt)
    df = pd.DataFrame({'cif_id':['Material1'], 'surface area (m^2/g)':['10.234']}).set_index('cif_id')
    
    output = chain.run(
        question = 'What is the highest bandgap MOF?',
        input='''```
max_bandgap = df['bandgap'].max()
max_mof = df[df['bandgap'] == max_bandgap]['cif_id'].values[0]
print(max_mof)
```''',
        observation='MAXHIE_clean',
    )
    print (output)
