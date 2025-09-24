import json
import os
from datetime import datetime

DB_FILE = "sentiment_store.json"

def store_sentiment(company, sentiment_result):
    if not os.path.exists(DB_FILE):
        db = {}
    else:
        with open(DB_FILE, "r") as f:
            db = json.load(f)

    if company not in db:
        db[company] = []

    db[company].append({
        "timestamp": datetime.utcnow().isoformat(),
        "sentiment": sentiment_result.sentiment,
        "score": sentiment_result.confidence_score
    })

    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)


def load_sentiment_history(company):
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        db = json.load(f)
    return db.get(company, [])
