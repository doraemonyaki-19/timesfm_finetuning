
import yfinance as yf
import pandas as pd
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sp500_tickers():
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    try:
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_df = pd.read_html(sp500_url)[0]
        sp500_tickers = sp500_df['Symbol'].tolist()
        # Replace dots with dashes for tickers like BRK.B -> BRK-B
        sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]
        logging.info(f"Successfully fetched {len(sp500_tickers)} S&P 500 tickers.")
        return sp500_tickers
    except Exception as e:
        logging.error(f"Could not fetch S&P 500 tickers: {e}")
        # Fallback list in case of network/parsing error
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']

def get_returns(tickers, period="5y"):
    """Downloads historical data and calculates daily returns."""
    try:
        data = yf.download(tickers, period=period, auto_adjust=True)['Close']
        returns = data.pct_change().dropna()
        # Filter out tickers with insufficient data
        valid_tickers = returns.columns[returns.notna().all()].tolist()
        invalid_tickers = set(tickers) - set(valid_tickers)
        if invalid_tickers:
            logging.warning(f"Could not fetch complete data for: {', '.join(invalid_tickers)}")
        return returns[valid_tickers]
    except Exception as e:
        logging.error(f"Error downloading data for tickers: {e}")
        return pd.DataFrame()

def find_correlated_tickers(target_tickers, candidate_tickers, top_n=5):
    """Finds the most correlated tickers from a candidate pool for each target ticker."""
    all_tickers = list(set(target_tickers) | set(candidate_tickers))
    logging.info(f"Fetching historical data for {len(all_tickers)} tickers...")
    returns = get_returns(all_tickers)
    
    # Ensure target tickers are in the returned data
    target_tickers = [t for t in target_tickers if t in returns.columns]
    if not target_tickers:
        logging.error("None of the target tickers have sufficient historical data.")
        return {}

    logging.info("Calculating correlation matrix...")
    corr_matrix = returns.corr()

    results = {}
    for target in target_tickers:
        # Get correlations for the target, drop self-correlation
        correlations = corr_matrix[target].drop(target, errors='ignore')
        # Filter to only include candidate tickers and remove any other target tickers
        candidate_correlations = correlations.loc[correlations.index.isin(candidate_tickers)]
        
        # Sort by correlation and get top N
        top_correlated = candidate_correlations.nlargest(top_n)
        results[target] = top_correlated
        logging.info(f"Top {top_n} correlated tickers for {target}:\n{top_correlated}\n")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Find top N correlated tickers for a given set of targets.")
    parser.add_argument("--targets", required=True, help="Comma-separated list of target tickers.")
    parser.add_argument("--top-n", type=int, default=2, help="Number of top correlated tickers to find for each target.")
    
    args = parser.parse_args()
    target_tickers = [t.strip().upper() for t in args.targets.split(',')]
    
    candidate_tickers = get_sp500_tickers()
    # Ensure candidates do not include targets
    candidate_tickers = [c for c in candidate_tickers if c not in target_tickers]

    find_correlated_tickers(target_tickers, candidate_tickers, top_n=args.top_n)

if __name__ == "__main__":
    main()
