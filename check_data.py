"""
Script to check if the data files contain fake or real data.
This helps users verify whether they need to run fetch_real_data.py
"""

import os
import pandas as pd
import sys

def check_data_files(data_dir='data'):
    """Check if data files exist and provide information about them."""
    
    print("="*60)
    print("Data Status Check")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        print("   Please run: python fetch_real_data.py")
        return False
    
    # Check for SPXconst first
    const_file = 'SPXconst.csv'
    
    # Look for Close and Open files with any year
    close_files = [f for f in os.listdir(data_dir) if f.startswith('Close-') and f.endswith('.csv')]
    open_files = [f for f in os.listdir(data_dir) if f.startswith('Open-') and f.endswith('.csv')]
    
    # Files to check - use any Close/Open file found, or default to 1990
    files_to_check = [const_file]
    if close_files:
        files_to_check.append(close_files[0])
    else:
        files_to_check.append('Close-1990.csv')
    
    if open_files:
        files_to_check.append(open_files[0])
    else:
        files_to_check.append('Open-1990.csv')
    
    all_exist = True
    real_data_indicators = []
    
    for filename in files_to_check:
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"\n✓ Found: {filename}")
            
            # Check file size and properties
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"  Size: {file_size:.1f} KB")
            
            # Try to read and analyze the file
            try:
                df = pd.read_csv(filepath)
                print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                
                # Check for indicators of real data
                if filename.startswith('Close') or filename.startswith('Open'):
                    # Check if using real ticker symbols (not S123 format)
                    sample_cols = list(df.columns[1:6])  # Sample first few columns
                    print(f"  Sample tickers: {sample_cols}")
                    
                    # Fake data uses format like S123, S456
                    fake_pattern = all(col.startswith('S') and col[1:].isdigit() 
                                     for col in sample_cols if col != 'Date')
                    
                    if fake_pattern:
                        print("  ⚠️  This appears to be FAKE data (S123 format)")
                        real_data_indicators.append(False)
                    else:
                        print("  ✅ This appears to be REAL data (actual tickers)")
                        real_data_indicators.append(True)
                        
                elif filename == 'SPXconst.csv':
                    # Check SPXconst
                    sample_stocks = list(df.iloc[0:3, 0])  # First few stocks in first month
                    print(f"  Sample stocks: {sample_stocks}")
                    
                    fake_pattern = all(str(s).startswith('S') and str(s)[1:].isdigit() 
                                     for s in sample_stocks if pd.notna(s))
                    
                    if fake_pattern:
                        print("  ⚠️  This appears to be FAKE data (S123 format)")
                        real_data_indicators.append(False)
                    else:
                        print("  ✅ This appears to be REAL data (actual tickers)")
                        real_data_indicators.append(True)
                        
            except Exception as e:
                print(f"  ❌ Error reading file: {str(e)}")
                
        else:
            print(f"\n❌ Missing: {filename}")
            all_exist = False
    
    print("\n" + "="*60)
    
    if not all_exist:
        print("Status: INCOMPLETE - Some data files are missing")
        print("\nTo fetch real data, run:")
        print("  python fetch_real_data.py --start_year 1990 --end_year 2018")
        return False
    
    if real_data_indicators and any(real_data_indicators):
        if all(real_data_indicators):
            print("Status: ✅ All data files contain REAL stock data")
            print("\nYou can now run the forecasting models:")
            print("  python NextDay-240,1-RF.py")
            print("  python Intraday-240,1-LSTM.py")
            print("  etc.")
        else:
            print("Status: ⚠️  Mixed - Some files are real, some are fake")
            print("\nRecommendation: Re-run the data fetching script:")
            print("  python fetch_real_data.py --start_year 1990 --end_year 2018")
    else:
        print("Status: ⚠️  All data files contain FAKE/DUMMY data")
        print("\nTo use REAL stock data, run:")
        print("  python fetch_real_data.py --start_year 1990 --end_year 2018 --backup_existing")
        print("\nThe dummy data is only for demonstration purposes.")
        print("For actual forecasting and research, you need real data.")
    
    print("="*60)
    
    return all_exist


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    check_data_files(data_dir)
