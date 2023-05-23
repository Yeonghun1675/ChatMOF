from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

#from langchain.load_model import MOFTransformer


def _get_predict_properties() -> BaseTool:
    return Tool(
        name="predictor",
        description="A tool that takes the path of a structure file (ex. *.cif) as input to predict the properties (ex. void fraction) of a mateirals.",
        func=lambda t: "Surface area : 20 m^2/m^3",
        #coroutine=PredictMofT().arun
    )