import yfinance as yf

sectors = {
    "Technology": ["AAPL", "MSFT"],
    "Healthcare": ["JNJ", "PFE"],
    "Financial Services": ["JPM", "BAC"],
    "Energy": ["XOM", "CVX"],
    "Consumer Defensive": ["PG", "WMT"],
    "Industrials": ["CAT", "HON"]
}

for sector, tickers in sectors.items():
    print(f"\n--- {sector} ---")
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        print(f"  {ticker}: {info.get('longName')}")
