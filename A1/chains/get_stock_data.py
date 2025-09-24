import yfinance as yf
from langchain_core.tools import tool
import mlflow

# import requests

# # Patch requests.Session.request to disable SSL verification
# original_request = requests.Session.request
# def unsafe_request(self, *args, **kwargs):
#     kwargs['verify'] = False
#     return original_request(self, *args, **kwargs)
# requests.Session.request = unsafe_request

@tool
def get_stock_ticker(company_name: str) -> str:
    """Returns the stock ticker using yfinance."""
    mlflow.set_tag("stage", "stock_code_extraction")
    search = yf.Ticker(company_name)
    if search:
        info = search.info
        return info.get("symbol", "")
    return ""
