import yfinance as yf
from langchain_core.tools import tool
import mlflow
from yahooquery import search

# import requests

def get_stock_ticker(company_name: str) -> str:
    result = search(company_name)
    # print(search)
    quotes = result.get("quotes", [])
    if not quotes:
        return None
    top_match = quotes[0]
    return top_match.get("symbol")

# @tool
# def get_stock_ticker(company_name: str) -> str:
#     """Returns the stock ticker using yfinance."""
#     mlflow.set_tag("stage", "stock_code_extraction")
#     search = yf.Ticker(company_name)
#     print(f"ticker search data: {search}")
#     if search:
#         info = search.info
#         print(f"info: {info}")
#         symbol = info.get("symbol", "")
#         print(f"symbol: {symbol}")
#         return info.get("symbol", "")
#     return ""
