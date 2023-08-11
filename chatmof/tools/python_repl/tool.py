from langchain.tools.base import BaseTool
from chatmof.tools.python_repl.base import PythonREPLTool


def _get_python_repl(**kwargs) -> BaseTool:
    return PythonREPLTool()