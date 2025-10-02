# Usage Guide

This guide explains how to use the stock market forecasting models with real data.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install scikit-learn==0.20.4 tensorflow==1.14.0
pip install yfinance pandas numpy
```

### 2. Fetch Real Data

**Important:** The default data files are fake/dummy data. To use real stock data:

```bash
python fetch_real_data.py --start_year 1990 --end_year 2018 --backup_existing
```

Options:
- `--start_year`: Starting year for data collection (default: 1990)
- `--end_year`: Ending year for data collection (default: 2018)
- `--output_dir`: Directory to save data files (default: data)
- `--backup_existing`: Backup existing fake data before overwriting

The script will:
- Download historical stock prices from Yahoo Finance
- Create `Close-YYYY.csv` with adjusted close prices
- Create `Open-YYYY.csv` with open prices
- Create `SPXconst.csv` with monthly S&P 500 constituents
- Save all files in the `data/` directory

**Note:** Fetching data for 1990-2018 may take 30-60 minutes depending on your internet connection.

### 3. Run Forecasting Models

Once you have real data, run any of the forecasting scripts:

#### Next Day Prediction Models

Predict next day's stock movements:

```bash
# Random Forest model
python NextDay-240,1-RF.py

# LSTM model
python NextDay-240,1-LSTM.py
```

#### Intraday Prediction Models

Predict intraday stock movements:

```bash
# Random Forest - 1 feature
python Intraday-240,1-RF.py

# LSTM - 1 feature
python Intraday-240,1-LSTM.py

# Random Forest - 3 features
python Intraday-240,3-RF.py

# LSTM - 3 features
python Intraday-240,3-LSTM.py
```

### 4. View Results

Results are saved in result directories:
- `results-NextDay-240-1-RF/`
- `results-NextDay-240-1-LSTM/`
- `results-Intraday-240-1-RF/`
- `results-Intraday-240-1-LSTM/`
- `results-Intraday-240-3-RF/`
- `results-Intraday-240-3-LSTM/`

Each result directory contains:
- `predictions-YYYY.pickle`: Model predictions for each year
- `avg_daily_rets-YYYY.csv`: Average daily returns
- `avg_returns.txt`: Summary statistics (mean returns and Sharpe ratio)

## Data Sources

### Yahoo Finance (Default)

The `fetch_real_data.py` script uses [yfinance](https://github.com/ranaroussi/yfinance) to download data from Yahoo Finance. This is:
- ✅ Free
- ✅ Easy to use
- ✅ No API key required
- ⚠️ May have occasional data gaps
- ⚠️ Historical constituent data is approximate

### Alternative Data Sources

For production or research use, consider:

1. **Bloomberg Terminal** (Used in the original paper)
   - Most comprehensive and accurate
   - Requires subscription
   
2. **Refinitiv/Thomson Reuters**
   - Professional-grade data
   - Requires subscription

3. **Quandl/Nasdaq Data Link**
   - Good quality historical data
   - Free tier available

4. **Alpha Vantage**
   - Free API with rate limits
   - Requires API key

To use alternative sources, modify `fetch_real_data.py` or manually create the data files following the format described in `data/README.md`.

## Troubleshooting

### "Module not found" errors

Install missing dependencies:
```bash
pip install -r requirements.txt
```

### "File not found" errors for data files

Run the data fetching script:
```bash
python fetch_real_data.py
```

### Data fetching is slow

This is normal. Downloading 20+ years of data for 300+ stocks takes time. The script shows progress as it runs.

### Some stocks failed to download

This is expected. Some tickers may be delisted or renamed. The script automatically handles this and uses stocks that were successfully downloaded.

### Out of memory errors

Try reducing the date range:
```bash
python fetch_real_data.py --start_year 2000 --end_year 2018
```

## Model Parameters

The models use these key parameters:

- **Lookback window**: 240 trading days (~1 year)
- **Training period**: 3 years rolling window
- **Test period**: 1 year
- **Random Forest**: 1000 trees, max depth 10-20
- **LSTM**: 25 cells, dropout 0.1

These parameters are based on the research paper and can be modified in the Python scripts.

## Citation

If you use this code or models in your research, please cite:

```bibtex
@article{ghosh2021forecasting,
  title={Forecasting directional movements of stock prices for intraday trading using LSTM and random forests},
  author={Ghosh, Pushpendu and Neufeld, Ariel and Sahoo, Jajati Keshari},
  journal={Finance Research Letters},
  pages={102280},
  year={2021},
  publisher={Elsevier}
}
```

Paper: https://arxiv.org/abs/2004.10178
