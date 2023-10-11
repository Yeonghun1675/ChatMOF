from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from chatmof.tools import load_chatmof_tools
from chatmof.agents.prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX


class ChatMOF(Chain):
    llm: BaseLanguageModel
    agent: object
    verbose: bool = False
    input_key: str = "query"
    output_key: str = "output"
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _call(
            self,
            inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        query = inputs[self.input_key]
        output = self.agent.run(query)
        
        return {
            self.output_key: output
        }
    
    @classmethod
    def from_llm(
        cls: BaseModel,
        llm: BaseLanguageModel,
        verbose: bool = False,
        search_internet: bool = True,
    ) -> Chain:
        
        tools = load_chatmof_tools(
            llm=llm, 
            verbose=verbose, 
            search_internet=search_internet
        )
        
        agent_kwargs = {
            'prefix': PREFIX,
            'format_instructions': FORMAT_INSTRUCTIONS,
            'suffix': SUFFIX
        }
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            agent_kwargs=agent_kwargs,
            #handle_parsing_errors=True,
        )

        print (type(agent))

        return cls(agent=agent, llm=llm, verbose=verbose)
