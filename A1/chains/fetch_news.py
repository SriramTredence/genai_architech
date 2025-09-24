from langchain_community.tools import BraveSearch
from langchain_core.tools import tool
import mlflow

# Assumes you've set up Brave API key in ENV
@tool
def fetch_news(company_name: str) -> str:
    """Fetches latest news headlines for the company."""
    mlflow.set_tag("stage", "news_fetch")
    search = BraveSearch()
    results = search.run(f"{company_name} stock news")
    return results[:5]  # return top 5 news
