DF_PROMPT = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should make a valid python command as input. You must use print the output using the `print` function at the end.
You should use the `to_markdown` function when you print a pandas object.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Input: the valid python code only using the Pandas library
Observation: the result of python code
... (this Thought/Input/Observation can repeat N times)
Final Thought: you should think about how to answer the question based on your observation
Final Answer: the final answer to the original input question. If you can't answer the question, say `nothing`

The index of the dataframe must be be one of {df_index}. If it's not in the index you want, skip straight to Final Thought.
{information}

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