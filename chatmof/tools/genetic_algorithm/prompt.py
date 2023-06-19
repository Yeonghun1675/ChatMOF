PLAN_PROMPT = (
    "Planning the generation task for two steps -> passs"
    """
    Question: ${{Generate problem}}
    Objective: ${{To solve the problem}}
    Search: ${{pass}}
    Genetic Algorithm: ${{genetic algoritm}}
    
    Begin.
    Question: Generate the void fraction with 
    Objective: pass
    Search: pass
    Genetic Algorithm: pass

    Question: {question}
    """
)

GENETIC_PROMPT = (
    "You should act as a generator to find the optimal material. "
    "A substance consists of a block1, block2, and must maintain the order. "
    "I will give you 200 parent materials with value. "
    "Based on these, you must answer as many new children as you expect to answer the question. "
    "You output children only and nothing else."
    "\n\n"
    "Begin.\n"
    "Question: {question}\n"
    "Parent:\n"
    "V1+T1 (value: 1.0)\n"
    "V2+T2 (value: 0.0)\n"
    "2 new Children:\n"
    "V2+T1, V1+T2\n"
    "Parent:\n"
    "{parents}"
    "200 new Children:\n"
)