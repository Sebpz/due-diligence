import yfinance as yf
import pandas as pd

def get_company_data(ticker_symbol: str) -> dict:
    """
    Retrieves key financial data for a given company ticker using yfinance.

    This function fetches the company's information and its historical
    stock price data for the last 5 days.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., 'AAPL' for Apple).

    Returns:
        A dictionary containing the company's info and its recent
        historical price data. Returns an empty dictionary if the ticker
        is invalid or no data is found.
    """
    try:
        # Create a Ticker object for the company
        company = yf.Ticker(ticker_symbol)

        # Get general company information
        # The .info dictionary contains various details like company name,
        # sector, market cap, etc.
        company_info = company.info

        # Get historical price data for the last 5 days
        # The period parameter can be changed to '1mo', '1y', 'max', etc.
        historical_data = company.history(period="5d")

        # Return a dictionary with the retrieved data
        return {
            "info": company_info,
            "historical_prices": historical_data
        }

    except Exception as e:
        print(f"Error retrieving data for {ticker_symbol}: {e}")
        return {}

# --- Example Usage ---
# Let's get data for Apple (AAPL)
apple_ticker = 'AAPL'
apple_data = get_company_data(apple_ticker)

# Check if data was retrieved successfully and print it
if apple_data:
    print(f"--- Data for {apple_ticker} ---")
    print("\nCompany Info:")
    # Print a few key pieces of information from the .info dictionary
    for key in ['longName', 'sector', 'marketCap', 'currentPrice']:
        if key in apple_data["info"]:
            print(f"  {key}: {apple_data['info'][key]}")

    print("\nHistorical Prices (last 5 days):")
    # Print the historical data, which is a pandas DataFrame
    print(apple_data["historical_prices"][['Open', 'Close', 'Volume']].to_string())
else:
    print(f"Failed to retrieve data for {apple_ticker}.")

# --- Another Example (Invalid Ticker) ---
print("\n" + "="*40 + "\n")
invalid_ticker = 'INVALIDTICKER'
invalid_data = get_company_data(invalid_ticker)
if not invalid_data:
    print(f"Correctly handled invalid ticker: {invalid_ticker}")
