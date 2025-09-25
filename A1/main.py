from chains.get_stock_data import get_stock_ticker
from chains.fetch_news import fetch_news
from chains.perform_sentiment_analysis import analyze_sentiment
from utils.logger import init_mlflow, start_run
import mlflow
import json
import argparse

def run_chain(company_name):
    init_mlflow()
    with start_run(company_name):
        mlflow.log_param("company_name", company_name)

        # Step 1: Extract stock symbol
        stock_code = get_stock_ticker(company_name)
        mlflow.log_param("stock_code", stock_code)

        # Step 2: Fetch news
        news = fetch_news(stock_code)
        mlflow.log_text(str(news), "fetched_news.txt")

        # Step 3: Sentiment Analysis
        result = analyze_sentiment(company_name, stock_code, news)
        mlflow.log_dict(result.dict(), "sentiment_output.json")

        print(json.dumps(result.dict(), indent=2))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", type=str, required=True, help="Company name (e.g., Microsoft)")
    args = parser.parse_args()

    run_chain(args.company)
