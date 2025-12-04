import yfinance as yf
import pandas as pd
import numpy as np

# 1. 設定參數
stock_id = "2330.TW"  # 您可以更改為任何台股代號
start_date = "2020-01-01"
entry_range_pct = 0.03   # 接近範圍 3%
profit_target_pct = 0.03 # 短線獲利目標 3%
stable_days = 5          # 站穩觀察天數
trend_period = 20        # 趨勢觀察天數 (一個月)

# 2. 取得資料 & 計算 200MA
df = yf.download(stock_id, start=start_date)
df['200MA'] = df['Close'].rolling(window=200).mean()

# 為了方便計算，移除沒有 200MA 的前期資料
df = df.dropna()

results = []

# 3. 逐日回測邏輯
# 我們遍歷每一天，尋找符合「進場條件」的日子
for i in range(len(df) - trend_period - stable_days):
    today = df.iloc[i]
    current_idx = df.index[i]
    
    close_price = today['Close']
    ma_price = today['200MA']
    
    # --- 條件 A：進場判斷 ---
    # 股價在 200MA 之上，且距離小於 3%
    if ma_price < close_price <= ma_price * (1 + entry_range_pct):
        
        # 取得未來幾天的資料
        future_5_days = df.iloc[i+1 : i+1+stable_days]
        future_month = df.iloc[i+1+stable_days : i+1+stable_days+trend_period]
        
        # --- 判斷 1：短線反彈獲利 (一周內是否漲 3%) ---
        # 檢查未來 5 天的最高價是否碰到 (進場價 * 1.03)
        target_price = close_price * (1 + profit_target_pct)
        is_short_term_win = future_5_days['High'].max() >= target_price
        
        # --- 判斷 2：是否站穩 (一周都沒跌破 200MA) ---
        # 檢查未來 5 天的收盤價是否都在當天的 200MA 之上 (簡化比較，或取動態 200MA 比較)
        # 嚴謹做法：比較每天的收盤價與當天的 200MA
        is_stable = True
        for j in range(stable_days):
            day_data = df.iloc[i+1+j]
            if day_data['Close'] < day_data['200MA']:
                is_stable = False
                break
        
        # 如果站穩，計算後面一個月的績效
        trend_return = None
        if is_stable:
            # 一個月後的收盤價 vs 站穩期結束時的收盤價
            price_after_stable = future_5_days.iloc[-1]['Close']
            price_after_month = future_month.iloc[-1]['Close']
            trend_return = (price_after_month - price_after_stable) / price_after_stable

        results.append({
            "Entry_Date": current_idx,
            "Entry_Price": close_price,
            "MA_Price": ma_price,
            "Short_Term_Win": is_short_term_win,
            "Stable_5_Days": is_stable,
            "Month_Return": trend_return
        })

# 4. 統計結果
results_df = pd.DataFrame(results)

if not results_df.empty:
    print(f"--- 回測標的: {stock_id} ---")
    print(f"總觸發進場次數: {len(results_df)}")
    
    # 短線勝率
    win_rate = results_df['Short_Term_Win'].mean() * 100
    print(f"短線反彈成功率 (3%): {win_rate:.2f}%")
    
    # 站穩後的表現
    stable_cases = results_df[results_df['Stable_5_Days'] == True]
    print(f"成功站穩 5 天次數: {len(stable_cases)}")
    
    if not stable_cases.empty:
        avg_trend_return = stable_cases['Month_Return'].mean() * 100
        win_trend_cases = stable_cases[stable_cases['Month_Return'] > 0]
        trend_win_rate = len(win_trend_cases) / len(stable_cases) * 100
        
        print(f"站穩後，下個月平均報酬: {avg_trend_return:.2f}%")
        print(f"站穩後，下個月上漲機率: {trend_win_rate:.2f}%")
else:
    print("在此期間內未觸發進場條件")
