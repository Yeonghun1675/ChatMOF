from langchain.llms import OpenAI, OpenAIChat
from langchain.agents import load_tools

from executors.agent_executor import load_agent_executor
from planners.chat_planner import load_chat_planner
from agent_executor import PlanAndExecute


llm = OpenAI(temperature=0)
tools = load_tools(
    tool_names = ['serpapi','llm-math'],
    llm = llm,
)

planner = load_chat_planner(
    llm = llm,
    verbose = True,
)

executor = load_agent_executor(
    llm = llm,
    tools = tools,
    verbose = True,
)

pne = PlanAndExecute(
    planner = planner,
    executer = executor,
)

pne(
    {
        #'input': 'What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?'
        'input': 'What Surface area for IRMOF-1?'
    }
)