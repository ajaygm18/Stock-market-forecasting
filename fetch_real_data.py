"""
Script to fetch real S&P 500 stock data from Yahoo Finance.
This replaces the dummy/fake data with real historical stock prices.

Usage:
    python fetch_real_data.py --start_year 1990 --end_year 2018

Requirements:
    pip install yfinance pandas numpy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# S&P 500 stock tickers (as of various time periods)
# This is a simplified list - in production, you'd want historical constituent data
SP500_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'NVDA', 'JPM', 'JNJ',
    'V', 'PG', 'XOM', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'CMCSA',
    'NFLX', 'VZ', 'INTC', 'PFE', 'T', 'KO', 'PEP', 'ABT', 'CSCO', 'MRK',
    'ABBV', 'AVGO', 'ORCL', 'CVX', 'ACN', 'NKE', 'TMO', 'CRM', 'MCD', 'COST',
    'LLY', 'DHR', 'MDT', 'WMT', 'TXN', 'NEE', 'UNP', 'BMY', 'QCOM', 'HON',
    'PM', 'UPS', 'LOW', 'RTX', 'IBM', 'AMGN', 'CVS', 'SBUX', 'LIN', 'BA',
    'GE', 'CAT', 'GS', 'ISRG', 'INTU', 'AMD', 'DE', 'BLK', 'MMM', 'AXP',
    'NOW', 'EL', 'ADI', 'GILD', 'BKNG', 'SYK', 'PLD', 'MDLZ', 'CI', 'TJX',
    'ADP', 'VRTX', 'REGN', 'ZTS', 'MO', 'CB', 'SO', 'DUK', 'MMC', 'SCHW',
    'BDX', 'TMUS', 'SLB', 'BSX', 'EQIX', 'USB', 'CSX', 'SPGI', 'CL', 'EOG',
    'NSC', 'PNC', 'ITW', 'AON', 'ETN', 'CME', 'APD', 'TGT', 'HUM', 'CCI',
    'MU', 'FISV', 'LRCX', 'SHW', 'WM', 'ICE', 'DG', 'F', 'GM', 'EMR',
    'NOC', 'TFC', 'PSA', 'GD', 'ATVI', 'FDX', 'MCO', 'DOW', 'D', 'COF',
    'MAR', 'PYPL', 'PGR', 'ROP', 'JCI', 'ILMN', 'AIG', 'ECL', 'SRE', 'KLAC',
    'LHX', 'APH', 'ADSK', 'AFL', 'ROST', 'FCX', 'AEP', 'CMG', 'MET', 'MSCI',
    'ADM', 'AJG', 'SPG', 'ALL', 'TEL', 'WMB', 'TRV', 'TT', 'KMB', 'PSX',
    'HCA', 'PCG', 'PCAR', 'ANET', 'DLR', 'EW', 'CARR', 'CTVA', 'NXPI', 'PRU',
    'SYY', 'O', 'PAYX', 'WELL', 'MSI', 'SNPS', 'FTNT', 'GIS', 'CDNS', 'AZO',
    'ORLY', 'IDXX', 'CTAS', 'AMT', 'OKE', 'KMI', 'HLT', 'YUM', 'HSY', 'RSG',
    'A', 'IQV', 'BIIB', 'MNST', 'MCHP', 'CTSH', 'EXC', 'DD', 'FAST', 'XEL',
    'DXCM', 'ED', 'BK', 'GPN', 'EA', 'CHTR', 'VRSK', 'KHC', 'STZ', 'AMP',
    'WEC', 'ES', 'CPRT', 'GLW', 'PPG', 'RMD', 'BKR', 'OTIS', 'APTV', 'KEYS',
    'LYB', 'EBAY', 'AWK', 'VICI', 'SBAC', 'ROK', 'CBRE', 'DHI', 'ENPH', 'HPQ',
    'ANSS', 'MTD', 'EXR', 'DFS', 'MKTX', 'EIX', 'ALGN', 'VMC', 'MPWR', 'TSCO',
    'WBA', 'HPE', 'VTR', 'FTV', 'TDG', 'ZBRA', 'DOV', 'WY', 'FE', 'CNP',
    'LUV', 'ARE', 'ETR', 'WBD', 'K', 'DAL', 'CDW', 'STT', 'PTC', 'FITB',
    'FRC', 'IFF', 'TTWO', 'TYL', 'BR', 'DTE', 'NTRS', 'TROW', 'AVB', 'LH',
    'CAH', 'PPL', 'GWW', 'HBAN', 'RF', 'CSGP', 'DRI', 'NTAP', 'SWK', 'EQR',
    'CMS', 'AEE', 'INVH', 'BALL', 'AKAM', 'CFG', 'NVR', 'MAS', 'LEN', 'UAL',
    'PFG', 'HOLX', 'VTRS', 'FLT', 'DGX', 'SYF', 'POOL', 'ESS', 'NI', 'EXPD',
    'MAA', 'TER', 'JBHT', 'KEY', 'ATO', 'LVS', 'BBY', 'PKI', 'EFX', 'EXPE',
    'CHD', 'WAT', 'IP', 'OMC', 'LDOS', 'KMX', 'NDAQ', 'UDR', 'JKHY', 'CE',
    'J', 'SWKS', 'MKC', 'CINF', 'TXT', 'HAL', 'CPT', 'AES', 'COO', 'HES',
    'EPAM', 'STE', 'WAB', 'HIG', 'CBOE', 'TECH', 'FDS', 'LNT', 'HST', 'BXP',
    'EMN', 'DPZ', 'AAL', 'PKG', 'MGM', 'ABMD', 'CRL', 'MLM', 'BBWI', 'VFC',
    'ULTA', 'PAYC', 'CHRW', 'LKQ', 'CF', 'WHR', 'TPR', 'ZION', 'CCL', 'IEX',
    'CTLT', 'BIO', 'NCLH', 'XRAY', 'PARA', 'AAP', 'NWL', 'HII', 'TAP', 'MOS',
    'BWA', 'FBHS', 'DVA', 'AIZ', 'PNW', 'UHS', 'RE', 'IPG', 'HSIC', 'SEE',
    'CZR', 'AOS', 'MTCH', 'DISH', 'HAS', 'IVZ', 'WRK', 'NWSA', 'RL', 'FMC',
    'ALK', 'DXC', 'NWS', 'ALLE', 'GL'
]


def fetch_stock_data(tickers, start_date, end_date, data_type='Close'):
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_type: 'Close' or 'Open'
    
    Returns:
        DataFrame with dates as rows and tickers as columns
    """
    print(f"Fetching {data_type} prices for {len(tickers)} stocks from {start_date} to {end_date}...")
    
    all_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        try:
            # Download data for this ticker
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty:
                # Use adjusted close for Close data, regular open for Open
                if data_type == 'Close':
                    all_data[ticker] = hist['Close']
                else:
                    all_data[ticker] = hist['Open']
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(tickers)} stocks...")
            else:
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"  Warning: Failed to fetch data for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"\nFailed to fetch data for {len(failed_tickers)} tickers: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
    
    # Combine into DataFrame
    df = pd.DataFrame(all_data)
    
    # Remove columns (tickers) with too many missing values
    threshold = len(df) * 0.5  # Keep tickers with at least 50% data
    df = df.dropna(axis=1, thresh=threshold)
    
    print(f"Successfully fetched data for {len(df.columns)} stocks with {len(df)} trading days")
    
    return df


def create_constituent_file(df, output_path):
    """
    Create SPXconst.csv file with monthly constituent lists.
    For simplicity, we use the same tickers for all months.
    
    Args:
        df: DataFrame with stock data
        output_path: Path to save the constituent file
    """
    print(f"\nCreating constituent file...")
    
    # Get all unique dates
    dates = df.index
    
    # Create monthly columns
    monthly_data = {}
    
    # Group by year-month
    df_with_month = df.copy()
    df_with_month['YearMonth'] = df_with_month.index.to_period('M')
    
    for year_month in df_with_month['YearMonth'].unique():
        # Format as MM/YYYY
        month_str = f"{year_month.month:02d}/{year_month.year}"
        
        # For each month, list all available tickers
        # In reality, this should reflect actual S&P 500 membership at that time
        # Here we use all tickers that have data in that month
        month_df = df_with_month[df_with_month['YearMonth'] == year_month]
        available_tickers = [col for col in month_df.columns if col != 'YearMonth' and not month_df[col].isna().all()]
        
        monthly_data[month_str] = available_tickers
    
    # Create DataFrame with equal-length columns (pad with NaN)
    max_len = max(len(v) for v in monthly_data.values())
    
    for key in monthly_data:
        monthly_data[key] = monthly_data[key] + [np.nan] * (max_len - len(monthly_data[key]))
    
    const_df = pd.DataFrame(monthly_data)
    const_df.to_csv(output_path, index=False)
    
    print(f"Saved constituent file to {output_path}")
    print(f"  {len(const_df.columns)} months with up to {len(const_df)} stocks per month")


def save_price_data(df, start_year, data_type, output_dir):
    """
    Save price data in the format expected by the forecasting scripts.
    
    Args:
        df: DataFrame with stock prices
        start_year: The starting year for the filename
        data_type: 'Close' or 'Open'
        output_dir: Directory to save the file
    """
    # Reset index to make Date a column
    df_to_save = df.reset_index()
    df_to_save.rename(columns={'index': 'Date', 'Date': 'Date'}, inplace=True)
    
    # Format dates as YYYY-MM-DD
    df_to_save['Date'] = pd.to_datetime(df_to_save['Date']).dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    output_path = os.path.join(output_dir, f'{data_type}-{start_year}.csv')
    df_to_save.to_csv(output_path, index=False)
    
    print(f"Saved {data_type} data to {output_path}")
    print(f"  Shape: {df_to_save.shape[0]} days x {df_to_save.shape[1]-1} stocks")


def main():
    parser = argparse.ArgumentParser(description='Fetch real S&P 500 stock data from Yahoo Finance')
    parser.add_argument('--start_year', type=int, default=1990, 
                        help='Start year for data collection (default: 1990)')
    parser.add_argument('--end_year', type=int, default=2018,
                        help='End year for data collection (default: 2018)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for data files (default: data)')
    parser.add_argument('--backup_existing', action='store_true',
                        help='Backup existing fake data before overwriting')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Backup existing fake data if requested
    if args.backup_existing:
        backup_dir = os.path.join(args.output_dir, 'fake_data_backup')
        os.makedirs(backup_dir, exist_ok=True)
        
        for filename in ['SPXconst.csv', f'Close-{args.start_year}.csv', f'Open-{args.start_year}.csv']:
            filepath = os.path.join(args.output_dir, filename)
            if os.path.exists(filepath):
                backup_path = os.path.join(backup_dir, filename)
                os.rename(filepath, backup_path)
                print(f"Backed up {filename} to {backup_dir}")
    
    # Set date range
    start_date = f'{args.start_year}-01-01'
    # Add extra years to ensure we have enough data for the rolling window
    # The scripts use test_year-3 for training data
    end_date = f'{args.end_year + 3}-12-31'
    
    print("="*60)
    print("Fetching Real S&P 500 Stock Data")
    print("="*60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of tickers: {len(SP500_TICKERS)}")
    print()
    
    # Fetch Close prices
    close_df = fetch_stock_data(SP500_TICKERS, start_date, end_date, data_type='Close')
    save_price_data(close_df, args.start_year, 'Close', args.output_dir)
    
    print()
    
    # Fetch Open prices
    open_df = fetch_stock_data(SP500_TICKERS, start_date, end_date, data_type='Open')
    save_price_data(open_df, args.start_year, 'Open', args.output_dir)
    
    print()
    
    # Create constituent file (using close data as reference)
    const_path = os.path.join(args.output_dir, 'SPXconst.csv')
    create_constituent_file(close_df, const_path)
    
    print()
    print("="*60)
    print("Data fetching complete!")
    print("="*60)
    print(f"\nYou can now run the forecasting scripts with real data.")
    print(f"Example: python NextDay-240,1-RF.py")


if __name__ == '__main__':
    main()
