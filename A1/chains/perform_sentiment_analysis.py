from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
import mlflow
import os
from dotenv import load_dotenv
load_dotenv()
class SentimentOutput(BaseModel):
    company_name: str
    stock_code: str
    newsdesc: str
    sentiment: str
    people_names: list[str]
    places_names: list[str]
    other_companies_referred: list[str]
    related_industries: list[str]
    market_implications: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)

parser = PydanticOutputParser(pydantic_object=SentimentOutput)

prompt = PromptTemplate(
    template="""
You are a financial analyst LLM. Based on the following news:

{news}

For the company {company_name} ({stock_code}), analyze the sentiment and extract:

- Sentiment (Positive/Negative/Neutral)
- People mentioned
- Places mentioned
- Other companies
- Industries
- Market implications
- Confidence score (0.0 to 1.0)

Respond ONLY in this JSON schema:

{format_instructions}
""",
    input_variables=["company_name", "stock_code", "news"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
# print(f"API KEY {os.getenv("AZURE_OPENAI_API_KEY")}")
llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_DEPLOYMENT"), 
    temperature=0.2,
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

chain = prompt|llm|parser

def analyze_sentiment(company_name, stock_code, news):
    mlflow.set_tag("stage", "sentiment_analysis")
    return chain.invoke({
        "company_name": company_name,
        "stock_code": stock_code,
        "news": news,
    })
