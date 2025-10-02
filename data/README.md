## Data Directory

This directory contains stock market data files needed for forecasting.

### Fetching Real Data

**To use real S&P 500 stock data instead of dummy data**, run the data fetching script:

```bash
# Install required packages
pip install yfinance pandas numpy

# Fetch real data from Yahoo Finance
python fetch_real_data.py --start_year 1990 --end_year 2018 --backup_existing
```

This will download historical stock prices from Yahoo Finance and create the necessary data files.

### File Format

#### Close-YYYY.csv and Open-YYYY.csv

These files contain adjusted close prices and open prices respectively.
- Each row represents a trading day
- First column: Date (YYYY-MM-DD format)
- Subsequent columns: Stock prices for different tickers
- File naming: "type-yyyy.csv", where type = {'Close', 'Open'} and yyyy is the start year of training period

Example:
```
Date,AAPL,MSFT,GOOGL,...
1990-01-02,1.33,2.45,3.21,...
1990-01-03,1.35,2.47,3.19,...
```

#### SPXconst.csv

A constituent file listing S&P 500 stocks for each month.
- Each column represents a month (MM/YYYY format)
- Each row contains a stock ticker that was part of the S&P 500 during that month
- Stock names must match the column names in Close-YYYY.csv and Open-YYYY.csv files

Example:
```
01/1990,02/1990,03/1990,...
AAPL,AAPL,AAPL,...
MSFT,MSFT,MSFT,...
```

### Default Dummy Data

**The default data files in this directory are not real and are only for demonstration.**

The dummy data is provided to familiarize users with the required file format. For actual forecasting and research, please fetch real data using the `fetch_real_data.py` script.

### Notes

- In the [paper](https://arxiv.org/abs/2004.10178), Bloomberg was used to retrieve adjusted stock prices
- The `fetch_real_data.py` script uses Yahoo Finance as a free alternative
- For production use, consider using premium data sources like Bloomberg, Refinitiv, or Quandl
