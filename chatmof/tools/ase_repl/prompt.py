ASE_PROMPT = """You are working with a ase library in Python. The name of the object `ase.atoms` is `atoms`.
You should make a valid python command as input. You must use print the output using the `print` function at the end.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Input: the valid python code only using the ase library
Observation: the result of python code
... (this Thought/Input/Observation can repeat N times)
Final Thought: you should think about how to answer the question based on your observation
Final Answer: the final answer to the original input question. If you can't answer the question, say `nothing`


Begin!

Question: What is the head of df? If you extracted successfully, derive 'success' as the final answer
Thought: To get the head of a DataFrame, we can use the pandas function head(), which will return the first N rows. By default, it returns the first 5 rows.
Input: 
``` 
import pandas as pd
import json
print(df.head().to_markdown())
```
Observation: {df_head}
Final Thought: The head() function in pandas provides the first 5 rows of the DataFrame. 
Final Answer: success


Question: {question}
{agent_scratchpad}
"""