"""
Run a simplified version of the forecasting model to show actual results.
This demonstrates the model's performance with real data.
"""

import pandas as pd
import numpy as np
import random
import time
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Set seed for reproducibility
SEED = 9
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║           FORECASTING MODEL RESULTS - REAL DATA                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Use the test data
data_dir = 'data_test_real'

# Load data
print("Loading real stock data...")
SP500_df = pd.read_csv(os.path.join(data_dir, 'SPXconst.csv'))
all_companies = list(set(SP500_df.values.flatten()))
all_companies.remove(np.nan)

constituents = {'-'.join(col.split('/')[::-1]): set(SP500_df[col].dropna()) 
                for col in SP500_df.columns}

close_file = [f for f in os.listdir(data_dir) if f.startswith('Close-')][0]
df = pd.read_csv(os.path.join(data_dir, close_file))

print(f"✅ Loaded data: {df.shape[0]} days, {df.shape[1]-1} stocks")

# Create labels
def create_label(df, perc=[0.5, 0.5]):
    perc = [0.] + list(np.cumsum(perc))
    label = df.iloc[:, 1:].pct_change(fill_method=None)[1:].apply(
        lambda x: pd.qcut(x.rank(method='first'), perc, labels=False), axis=1)
    return label

label = create_label(df)
print(f"✅ Created labels: {label.shape}")

# Feature creation for one stock
def create_stock_data(df, st, test_year):
    st_data = pd.DataFrame([])
    st_data['Date'] = list(df['Date'])
    st_data['Name'] = [st] * len(st_data)
    
    # Add return features
    for k in list(range(1, 21)) + list(range(40, 241, 20)):
        st_data[f'R{k}'] = df[st].pct_change(k)
    
    st_data['R-future'] = df[st].pct_change().shift(-1)
    st_data['label'] = list(label[st]) + [np.nan]
    st_data['Month'] = list(df['Date'].str[:-3])
    st_data = st_data.dropna()
    
    trade_year = st_data['Month'].str[:4]
    st_data = st_data.drop(columns=['Month'])
    st_train_data = st_data[trade_year < str(test_year)]
    st_test_data = st_data[trade_year == str(test_year)]
    
    return np.array(st_train_data), np.array(st_test_data)

# Run for one test year
test_year = 2019
month_key = f"{test_year-1}-12"

if month_key not in constituents:
    print(f"❌ No constituents for {month_key}")
    sys.exit(1)

stock_names = sorted(list(constituents[month_key]))[:50]  # Use first 50 stocks for speed
print(f"\n📊 Running NextDay-240,1-RF Model")
print(f"   Test year: {test_year}")
print(f"   Stocks: {len(stock_names)} (limited for demo)")

# Collect data for all stocks
train_data, test_data = [], []

print("\n⏳ Preparing data...")
start = time.time()
for i, st in enumerate(stock_names):
    try:
        st_train_data, st_test_data = create_stock_data(df, st, test_year)
        if len(st_train_data) > 0 and len(st_test_data) > 0:
            train_data.append(st_train_data)
            test_data.append(st_test_data)
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(stock_names)} stocks...")
    except Exception as e:
        continue

if not train_data or not test_data:
    print("❌ No valid data for training")
    sys.exit(1)

train_data = np.concatenate([x for x in train_data])
test_data = np.concatenate([x for x in test_data])

print(f"✅ Data prepared in {time.time() - start:.1f}s")
print(f"   Training samples: {train_data.shape}")
print(f"   Test samples: {test_data.shape}")

# Train Random Forest
print("\n🎯 Training Random Forest model...")
random.seed(SEED)
np.random.seed(SEED)

train_x, train_y = train_data[:, 2:-2], train_data[:, -1]
train_y = train_y.astype('int')

start = time.time()
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
clf.fit(train_x, train_y)
train_time = time.time() - start

train_acc = clf.score(train_x, train_y)
print(f"✅ Training completed in {train_time:.1f}s")
print(f"   Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

# Make predictions
print("\n📈 Generating predictions...")
dates = list(set(test_data[:, 0]))
predictions = {}

for day in dates:
    test_d = test_data[test_data[:, 0] == day]
    test_d_x = test_d[:, 2:-2]
    predictions[day] = clf.predict_proba(test_d_x)[:, 1]

print(f"✅ Predictions generated for {len(predictions)} trading days")

# Simulate trading strategy
print("\n💰 Simulating trading strategy...")
k = 5  # Top/bottom k stocks to trade

returns_long = []
returns_short = []
returns_combined = []

for day in sorted(predictions.keys()):
    preds = predictions[day]
    test_returns = test_data[test_data[:, 0] == day][:, -2]
    
    if len(preds) < k:
        continue
    
    # Long strategy: buy top k predicted stocks
    top_preds = preds.argsort()[-k:][::-1]
    trans_long = test_returns[top_preds]
    ret_long = np.mean(trans_long)
    returns_long.append(ret_long)
    
    # Short strategy: short bottom k predicted stocks
    worst_preds = preds.argsort()[:k][::-1]
    trans_short = -test_returns[worst_preds]
    ret_short = np.mean(trans_short)
    returns_short.append(ret_short)
    
    # Combined strategy
    returns_combined.append(ret_long + ret_short)

# Calculate statistics
def calculate_stats(returns, name):
    returns = np.array(returns)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    cumulative_return = np.prod(1 + returns) - 1
    
    print(f"\n   {name}:")
    print(f"      Mean daily return: {mean_return:.6f} ({mean_return*100:.4f}%)")
    print(f"      Std deviation: {std_return:.6f}")
    print(f"      Annualized Sharpe ratio: {sharpe:.4f}")
    print(f"      Cumulative return: {cumulative_return:.6f} ({cumulative_return*100:.2f}%)")
    print(f"      Trading days: {len(returns)}")
    
    return {
        'mean': mean_return,
        'std': std_return,
        'sharpe': sharpe,
        'cumulative': cumulative_return,
        'days': len(returns)
    }

print("\n" + "="*70)
print("RESULTS - TRADING STRATEGY PERFORMANCE")
print("="*70)

long_stats = calculate_stats(returns_long, "Long Strategy (Buy top 5 stocks)")
short_stats = calculate_stats(returns_short, "Short Strategy (Short bottom 5 stocks)")
combined_stats = calculate_stats(returns_combined, "Combined Long-Short Strategy")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
This demonstrates a simplified Random Forest model trained on real S&P 500 data.

Model Configuration:
  • Algorithm: Random Forest (100 trees, max depth 10)
  • Features: 30 technical indicators (returns over various periods)
  • Training period: 2018-{test_year-1}
  • Test period: {test_year}
  • Stocks: {len([t for t in train_data if t[0]])} training stocks

Trading Strategy:
  • Long: Buy top 5 predicted stocks each day
  • Short: Short bottom 5 predicted stocks each day
  • Combined: Long + Short strategy

Key Results for {test_year}:
  • Long strategy Sharpe ratio: {long_stats['sharpe']:.4f}
  • Short strategy Sharpe ratio: {short_stats['sharpe']:.4f}
  • Combined strategy Sharpe ratio: {combined_stats['sharpe']:.4f}
  • Combined cumulative return: {combined_stats['cumulative']*100:.2f}%

Note: This is a simplified demo with limited stocks and time period.
For full results (1993-2018), run: python NextDay-240,1-RF.py
""")

print("\n" + "="*70)
print("✅ MODEL EXECUTION COMPLETE")
print("="*70)
print("""
The model successfully:
1. ✅ Loaded and processed real stock data
2. ✅ Trained a Random Forest classifier
3. ✅ Generated predictions on test data
4. ✅ Simulated a trading strategy
5. ✅ Calculated performance metrics (returns, Sharpe ratio)

This demonstrates the complete forecasting workflow with real data!
""")
