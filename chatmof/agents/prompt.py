PROMPT = (
    #"When you get a query from HUMAN, you have to use various tools to derive an answer. "
    #"In the Thought phase, you must check if you can make a final answer with the results so far."
    #"If the observation can answer the query, you must immediately give the final answer without further steps."

    #"\n"
    #"There are two tools you can use to obtain properties when creating plan steps. "
    #"Tool must be run separately for each property. "
    #"The two tools are prioritized and described below:\n"
    #"1. search_csv : search in a look-up table. Search for previously calculated data for CoREMOF datasets "
    #'The question must be written to produce a answer with the fewest number of tokens. \n'
    #"2. predictor : prediction using a machine learning model. \n"
    #"(e.g., irmof1) as input to predict the properties of a material. "
    #"Structure files are existed in directory \'database/structures/raw\'. "
    #"If you can't find the structure file, you must ask HUMAN for the path of the structure file. "
    #"Predictable properties: surface area, pore volume, hydrogen uptake, bandgap.\n"
    #"\n"
    #"HUMAN : {query}"
    "{query}\n"
    "Tool must be run separately for each property. "
    "Use [search_csv, predictor] tools to obtain properties, you must run the tool separately for each property."   
)