import os

__version__ = '0.1.6'   
__root_dir__ = os.path.dirname(__file__)


from langchain.chat_models import ChatOpenAI
from chatmof.agents.agent import ChatMOF
from chatmof.config import config


__all__ = [
    "ChatMOF",
    "config",
    "ChatOpenAI",
    "__version__",
]
