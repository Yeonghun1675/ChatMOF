PROMPT = """plan to visualize the structure. To answer the question, you have to fill in the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Material: names of materials separated using comma. If you need to proceed for all material, write *. To proceed for a specific topology, append the topology name with an * (ex. pcu*)

Begin!

Question: Visualize the ZVBEFJ and acs+N123+E132.
Thought: I need to visualize the structure ZVBEFJ and acs+N123+E132.
Material: ZVBEFJ, acs+N123+E132

Question: Visualize the all mofs with pcu topology
Thought: I need to visualize the structures with pcu topology.
Material: pcu*

Question: {question}"""