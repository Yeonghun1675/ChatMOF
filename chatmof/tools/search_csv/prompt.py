DF_PROMPT = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
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

Question: What is the surface area of material "AVAVIJ"?
Thought: I need to extract the "surface area" property from the dataframe and return it in JSON format.
Input: 
``` 
import pandas as pd
import json
row = df.loc[df['name'] == 'AVAVIJ']
value = row['Accessible Surface Area (m^2/g)'].values[0]
result = {{\"name\": \"AVAVIJ\", \"property\": \"Accessible Surface Area\", \"value\": value, \"unit\": \"m^2/g\"}}
print (json.dumps(result))
```
Final Thought: There are no data for AVAVIJ in the database
Final Answer: nothing

Question: {question}
{agent_scratchpad}
"""