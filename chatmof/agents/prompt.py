PROMPT = (
    "When you get a query from HUMAN, you have to use various tools to derive an answer. "
    "In the Thought phase, you must check if you can make a final answer with the results so far."
    "If the observation can answer the query, you must immediately give the final answer without further steps."
    #"You must not check your answer when you are possible to answer the question."

    "\n"
    "There are two tools you can use to obtain properties when creating plan steps. "
    "The two tools are prioritized and described below:\n"
    "1. search in a look-up table. Search for previously calculated data for CoREMOF datasets "
    "(starting with 6 capitalized English letters, e.g. ZUTSIM, ARAZAT_clean).\n"
    'The question must be written to produce a answer with the fewest number of tokens. '
    'List of properties: surface area, pore volume'
    "2. prediction using a machine learning model: you can take the path of a structure file "
    "(e.g., irmof1.cif) as input to predict the properties of a material. "
    "Structure files are existed in directory \'database/structures/raw\'. "
    "If you can't find the structure file, you must ask HUMAN for the path of the structure file. "
    "Predictable properties: surface area, pore volume, hydrogen uptake, bandgap.\n"
    "\n"
    "HUMAN : {query}"
)