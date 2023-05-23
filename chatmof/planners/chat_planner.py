import re

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from chatmof.planners.base import LLMPlanner
from chatmof.schema import (
    Plan,
    PlanOutputParser,
    Step,
)
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

SYSTEM_PROMPT = (
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "to accurately complete the task. If the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "At the end of your plan, say '<END_OF_PLAN>'\n"

    "There are three tools you can utilize when creating a plan:\n"
    "1. search in a look-up table. Search for previously calculated data for CoREMOF datasets "
    "(starting with 6 capitalized English letters, e.g. ZUTSIM).\n"
    "2. prediction using a machine learning model: you can take the path of a structure file "
    "(e.g., irmof1.cif) as input to predict the properties of a material. "
    "Predictable properties: surface area, pore volume, hydrogen uptake.\n"
    "3. Get the properties of a material using an internet search\n"
)


class PlanningOutputParser(PlanOutputParser):
    def parse(self, text: str) -> Plan:
        steps = [Step(value=v) for v in re.split("\n\d+\. ", text)[1:]]
        return Plan(steps=steps)


def load_chat_planner(
    llm: BaseLanguageModel, system_prompt: str = SYSTEM_PROMPT,  verbose: bool = False
) -> LLMPlanner:
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)
    return LLMPlanner(
        llm_chain=llm_chain,
        output_parser=PlanningOutputParser(),
        stop=["<END_OF_PLAN>"],
        verbose=verbose,
    )
