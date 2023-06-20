PROMPT = """plan to use machine learning to predict the properties of matter. To answer the question, you have to fill in the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Property: the property you can predict, should be one of [{tool_names}]
Material: names of materials separated using comma. If you need to proceed for all material, write *. To proceed for a specific topology, append the topology name with an * (ex. pcu*)
... (this Property/Material can repeat N times)
Run Machine Learning: nothing to do
Final Thought: get a final answer based on the values you found.

Begin!

Question: predict the surface area and hydrogen uptake of ACOPEN and ELOBPE.
Thought: I need to gather data on ACOPEN and train a machine learning model to predict its surface area and hydrogen uptake.
Property: surface_area
Material: ACOPEN, ELOBPE
Property: hydrogen_uptake_at_100bar
Material: ACOPEN, ELOBPE
Final Thought: Based on the trained model, answer the question using predicted surface area and the predicted hydrogen uptake at 100 bar.

Question: which MOF has a highest band-gap?
Thought: I need to gather data on the band-gap of different structures and compare them to find the one with the highest value.
Property: bandgap
Material: *
Final Thought: Based on the trained model, identify the structure with the highest predicted band-gap value.

Question: Predict surface area and save results in csv format, only pcu topology.
Thought: I need to gather data on the surface area of materials with pcu topology and train a machine learning model to predict their surface area. Then, I can use the model to make predictions and save the results in a csv format.
Property: surface_area
Material: pcu*
Final Thought: The predicted surface area values for materials with pcu topology can be found in the saved csv file.

Question: {question}"""


if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0.0)
    prompt = PromptTemplate(template=PROMPT, input_variables=['question', 'tool_names'])
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(
        question='Predict surface area and save results in csv format, only pcu topology.',
        tool_names=tool_names)
    print (output)