PLAN_PROMPT = """Create a plan to generate material based on the following question. 
Use the following format:

Question: the input question you must to answer
Thought: you should always think about what to do
Property: the property you can predict, must be one of [{model_names}]
Objective: you should decide what criteria you wan to generate by.
Search look-up table: plan to extract 100 material for the purpose from the look-up table where the property is pre-calculated.
Genetic algorithm: plan to create a new materials using the 100 extracted materials.
Final thought: get a final answer based on the structures you generate.

Begin!
Question: generate a material with a porosity of 0.5 and surface area of 120 m2/g
Thought: I need to generate a material with a porosity value of 0.5 and surface area of 120 m2/g. 
Property: void_fraction, accessible_surface_area
Objective: near 0.5, near 120
Search look-up table: extract cif_id, void_fraction, and accessible_surface_area of 100 materials with porosity close to 0.5 and surface area near 120 m2/g from look-up tables. 
Genetic algorithm: create a new material with a porosity close to 0.5 and surface area near 120 m2/g from 100 materials
Final Thought: Based on the generated CIF, find the material that is closest to a porosity of 0.5 and a surface area of 120 m2/g.

Question: generate a material with a highest band-gap
Thought: I need to generate a material with a highest band-gap.
Property: bandgap
Objective: max
Search look-up table: extract cif_id and bandgap of 100 materials with high band-gap value from look-up tables. 
Genetic algorithm: generate 100 new materials with the highest band gap from the 100 materials.
Final Thought: Based on the generated CIF, find the material that has highest band-gap.

Question: {question}"""


GENETIC_PROMPT = (
    "You should act as a generator to find the optimal material. "
    "A substance consists of a block1, block2, and must maintain the order. "
    "I will give you 100 parent materials. "
    "Based on these, you must answer as many new children as you expect to answer the question. "
    "The block1 and block2 used in the child must be the blocks used in the parent, and you must not create blocks that don't exist. "
    "You must generate children diverse. "
    "The children must not be duplicates of existing parents or already created children. "
    "You output children only and nothing else."
    "\n\n"
    "Begin.\n"
    "Question: {question}\n"
    "Parent:\n"
    "V12+T31, V24+T32, V7+T12\n"
    "4 new Children:\n"
    "V12+T12, V24+T31, V7+T31, V7+T32\n\n"
    "Parent:\n"
    "{parents}"
    "100 new Children:\n"
)