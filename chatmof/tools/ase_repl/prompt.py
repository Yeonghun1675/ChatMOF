ASE_PROMPT = """You are working with a ase library in Python. The name of the object `ase.atoms` is `atoms`.
You should make a valid python command as input. You must use print the output using the `print` function at the end.

Use the following format:

Question: the input question you must answer
Material: name of material
Convert: convert material to atoms object.
Thought: you should always think about what to do
Input: the valid python code only using the ase library
Observation: the result of python code
... (this Thought/Input/Observation can repeat N times)
Final Thought: you should think about how to answer the question based on your observation
Final Answer: the final answer to the original input question. If you can't answer the question, say `nothing`


Begin!

Question: calculate the cell volume of ABAYOU
Material: ABAYOU
Convert: Convert material to atoms object. I now have to write a code using `atoms` object
Thought: To calculate the cell volume of `atoms`, I need to access the cell information of the atoms object and then use the appropriate method to calculate its volume.
Input: 
``` 
import ase
volume = atoms.get_volume()
print(volume)
```
Observation: 18921.03834202299
Final Thought: The volume of the cell associated with the ase.atoms object atoms was successfully calculated using the get_volume() method from the ASE library.
Final Answer: The cell volume of the ABAYOU is 18921.03834202299 cubic Ångströms.

Question: {question}
{agent_scratchpad}
"""