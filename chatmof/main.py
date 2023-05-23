from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from executors.agent_executor import load_agent_executor
from planners.chat_planner import load_chat_planner
from agent_excutor import ChatMOF
from chatmof.planners.base import LLMPlanner
from chatmof.tools import load_mofchat_tools


llm = ChatOpenAI(temperature=0)
tool = load_mofchat_tools(llm)
#tool = load_tools(['serpapi', 'llm-math'], llm=llm)
verbose = True

planner = load_chat_planner(
    llm=llm,
    verbose=verbose,
)

executor = load_agent_executor(
    llm=llm,
    tools=tool,
    verbose=verbose,
)

assert isinstance(planner, LLMPlanner), f"{type(planner)}"


chatmof = ChatMOF(
    planner = planner,
    executer = executor,
)

chatmof(
    {
        'input': 'What is surface area of ACOGEF_clean?'
    }
)
