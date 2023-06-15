PROMPT = (
    "When you get a query from a HUMAN, you should do inverse design the material that corresponds to the question. "
    "For inverse design, you will proceed with a genetic algorithm. "
    "The structure is organized as follows: topology + building block1 + building block2 "
    "You must propose that combination to solve the problem using a genetic algorithm."

    "A genetic algorithm creates a new substance from pre-calculated data using two steps :\n"

    "1. crossover : Apply the crossover operation to the selected parents to create \"offspring\". "
    "This involves swapping parts of the parent individuals' data with each other. "
    "The point at which the data is swapped is chosen randomly. "
    "Example : acs+N23+E12, dia+N131+E6 -> acs+N131+E6 \n"
    "2. Mutation : With a 5% probability, randomly alter parts of the individuals in the population. "
    "This introduces new genetic material into the population and aids in maintaining diversity. "
    "The mutation must occur at a position in a pre-existing gene. "
    "Example : acs+N23+E12 -> acs+N31+E12 \n"

    "You make a plan for how to solve the query and perform a genetic algorithm based on that plan. "
    #"output_schema : {'Plan': str, 'Output': str}\n"
    #"example of output : \n"
    #"{'Plan': 'To achieve high uptake, we mutate genes with high uptake values, "
    #"'Output': 'acs+N1+E1, acs+N2+E2, acs+N3+E3, acs+N4+E4, pcu+N5+E6'}\n"
    "\n"
    "parents:\n"
    "{{ seed }}"
    "\n"
    "HUMAN: {{ query }} "
    "You must make  child gene with following format: \n"
    "Parent Selection: parent1, parent2\n"
    "Crossover: child\n"
    "Muation: Yes or No"
)


import math
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(PROMPT, template_format='jinja2')

seed = ''
for i in range(40):
    seed += f'acs+N{i}+E{i+1} = {math.cos(2*i)}\n'

output = prompt.format_prompt(
    seed=seed,
    query= 'Generate the structure with void fraction = 0.7',
)
print (output.text)