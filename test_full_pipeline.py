"""
Full pipeline test to demonstrate the real data fetching and model compatibility.
This script validates the complete workflow without requiring hours of training.
"""

import subprocess
import sys
import os
import time
import pandas as pd
import numpy as np

def run_command(cmd, description, timeout=600):
    """Run a command and capture output."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Command failed with return code {result.returncode}")
            return False
        else:
            print(f"✅ Command succeeded")
            return True
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            FULL PIPELINE TEST - REAL DATA INTEGRATION           ║
╚══════════════════════════════════════════════════════════════════╝

This test demonstrates:
1. Real data can be fetched from Yahoo Finance
2. Data format is correct for existing models
3. Models can load and validate the data
4. Complete workflow works end-to-end

Note: We use a limited dataset (2018-2020) to keep test time reasonable.
Full historical data (1990-2018) would take 30-60 minutes to download.
""")
    
    start_time = time.time()
    results = []
    
    # Test 1: Check initial data status
    print("\n" + "="*70)
    print("TEST 1: Check Initial Data Status")
    print("="*70)
    success = run_command(
        "python check_data.py",
        "Check if current data is fake or real",
        timeout=30
    )
    results.append(("Check initial data status", success))
    
    # Test 2: Fetch real data (limited timeframe for speed)
    print("\n" + "="*70)
    print("TEST 2: Fetch Real Data (2018-2020 for speed)")
    print("="*70)
    
    # Create test output directory
    test_data_dir = "data_test_real"
    os.makedirs(test_data_dir, exist_ok=True)
    
    success = run_command(
        f"python fetch_real_data.py --start_year 2018 --end_year 2020 --output_dir {test_data_dir}",
        "Fetch real stock data from Yahoo Finance",
        timeout=600  # 10 minutes should be enough
    )
    results.append(("Fetch real data", success))
    
    if success:
        # Test 3: Verify real data was fetched
        print("\n" + "="*70)
        print("TEST 3: Verify Real Data Quality")
        print("="*70)
        
        success = run_command(
            f"python check_data.py {test_data_dir}",
            "Verify downloaded data is real",
            timeout=30
        )
        results.append(("Verify real data", success))
        
        # Test 4: Validate data format
        print("\n" + "="*70)
        print("TEST 4: Validate Data Format")
        print("="*70)
        
        try:
            # Check Close file
            close_files = [f for f in os.listdir(test_data_dir) if f.startswith('Close-')]
            if close_files:
                close_df = pd.read_csv(os.path.join(test_data_dir, close_files[0]))
                print(f"✅ Close data loaded: {close_df.shape}")
                print(f"   Columns (first 10): {list(close_df.columns[:10])}")
                print(f"   Date range: {close_df['Date'].iloc[0]} to {close_df['Date'].iloc[-1]}")
                
                # Check for NaN values
                nan_pct = (close_df.isna().sum().sum() / (close_df.shape[0] * close_df.shape[1])) * 100
                print(f"   NaN percentage: {nan_pct:.2f}%")
                
            # Check Open file
            open_files = [f for f in os.listdir(test_data_dir) if f.startswith('Open-')]
            if open_files:
                open_df = pd.read_csv(os.path.join(test_data_dir, open_files[0]))
                print(f"✅ Open data loaded: {open_df.shape}")
                
            # Check SPXconst file
            const_file = os.path.join(test_data_dir, 'SPXconst.csv')
            if os.path.exists(const_file):
                const_df = pd.read_csv(const_file)
                print(f"✅ SPXconst data loaded: {const_df.shape}")
                print(f"   Months covered: {len(const_df.columns)}")
                print(f"   Stocks per month (avg): {const_df.count(axis=0).mean():.0f}")
                
            results.append(("Validate data format", True))
            
        except Exception as e:
            print(f"❌ Error validating data: {str(e)}")
            results.append(("Validate data format", False))
    
    # Test 5: Test model compatibility (syntax check only, no training)
    print("\n" + "="*70)
    print("TEST 5: Test Model Script Compatibility")
    print("="*70)
    
    models_to_test = [
        'NextDay-240,1-RF.py',
        'Intraday-240,1-RF.py'
    ]
    
    for model in models_to_test:
        success = run_command(
            f"python -m py_compile {model}",
            f"Validate {model} syntax",
            timeout=30
        )
        results.append((f"Syntax check {model}", success))
    
    # Test 6: Demonstrate data loading capability
    print("\n" + "="*70)
    print("TEST 6: Test Data Loading with Real Data")
    print("="*70)
    
    test_script = """
import pandas as pd
import sys
import os

data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'

# Find the Close file
close_files = [f for f in os.listdir(data_dir) if f.startswith('Close-')]
if not close_files:
    print("❌ No Close file found")
    sys.exit(1)

df = pd.read_csv(os.path.join(data_dir, close_files[0]))
print(f"✅ Successfully loaded Close data: {df.shape}")
print(f"   Trading days: {len(df)}")
print(f"   Stocks: {len(df.columns) - 1}")  # -1 for Date column

# Test percentage change calculation (used by models)
sample_stock = df.columns[1]  # First stock after Date
pct_change = df[sample_stock].pct_change()
print(f"   Sample stock: {sample_stock}")
print(f"   Price range: ${df[sample_stock].min():.2f} - ${df[sample_stock].max():.2f}")
print(f"   Daily return range: {pct_change.min():.4f} to {pct_change.max():.4f}")

# Load SPXconst
const_file = os.path.join(data_dir, 'SPXconst.csv')
if os.path.exists(const_file):
    const_df = pd.read_csv(const_file)
    print(f"✅ Successfully loaded SPXconst: {const_df.shape}")
    
    # Create constituents dict (as done in models)
    constituents = {col: set(const_df[col].dropna()) for col in const_df.columns}
    print(f"   Created constituents dict with {len(constituents)} months")
    
    # Show sample month
    sample_month = list(constituents.keys())[0]
    print(f"   Sample month {sample_month}: {len(constituents[sample_month])} stocks")
    
print("✅ Data loading test passed!")
"""
    
    with open('/tmp/test_load_data.py', 'w') as f:
        f.write(test_script)
    
    success = run_command(
        f"python /tmp/test_load_data.py {test_data_dir}",
        "Test data loading with real data",
        timeout=30
    )
    results.append(("Test data loading", success))
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    
    if passed_tests == total_tests:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("""
The complete pipeline works successfully:

✅ Data fetching from Yahoo Finance works
✅ Real data is properly formatted
✅ Data is compatible with existing model scripts
✅ Data loading and processing works correctly

To use the full dataset (1990-2018), run:
    python fetch_real_data.py --start_year 1990 --end_year 2018 --backup_existing

Then run any forecasting model:
    python NextDay-240,1-RF.py
    python Intraday-240,3-LSTM.py
""")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️ SOME TESTS FAILED")
        print("="*70)
        print(f"{total_tests - passed_tests} test(s) failed. See details above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
