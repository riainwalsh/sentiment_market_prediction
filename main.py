
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import yfinance as yf

def sample_headlines():
    # placeholder headlines
    texts = [
        "Company beats earnings expectations",
        "CEO resigns amid scandal",
        "Product launch receives positive reviews",
        "Regulatory probe weighs on shares",
        "Analysts upgrade stock to buy",
        "Supply chain concerns hit outlook",
        "Strong revenue growth drives rally",
        "Lawsuit risks dampen sentiment",
    ]
    labels = [1,0,1,0,1,0,1,0]  # simplistic positive(1)/negative(0)
    return texts, labels

def build_model(texts, labels):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=5000)
    X = vec.fit_transform(texts)
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, random_state=42, stratify=labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    print(classification_report(yte, preds))
    return vec, clf

def correlate_with_returns(ticker="AAPL", days=5):
    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
    rets = df["Close"].pct_change().dropna()
    avg = rets.tail(days).mean()
    return float(avg)

def main(ticker):
    texts, labels = sample_headlines()
    vec, clf = build_model(texts, labels)
    avg_ret = correlate_with_returns(ticker)
    print(f"Average {days:=5} day return (recent): {avg_ret:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()
    main(args.ticker)
