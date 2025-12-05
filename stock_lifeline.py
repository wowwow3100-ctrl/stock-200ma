import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime
import plotly.graph_objects as go
import requests
import os

# --- 1. 網頁設定 (必須放在最上面) ---
VER = "ver3.16 (Stable)"
st.set_page_config(page_title=f"🍍 旺來-台股生命線({VER})", layout="wide")

# --- CSS 優化: 加大字體與手機適配 ---
st.markdown("""
    <style>
    /* 全域字體加大 */
    html, body, [class*="css"]  {
        font-family: '微軟正黑體', sans-serif;
    }
    
    /* 表格字體優化 */
    .stDataFrame {
        font-size: 1.1rem !important;
    }
    
    /* 指標卡片字體加大 */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    
    /* 讓手機版表格好滑動 */
    .stDataFrame div[data-testid="stTable"] {
        overflow-x: auto;
    }
    
    /* 台股紅綠色定義 */
    .up-color { color: #ff4b4b !important; font-weight: bold; }
    .down-color { color: #00cc96 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 2. 核心功能區 ---

# ★★★ 關鍵修正：加入 show_spinner=False 避免 Streamlit 3.13 執行緒錯誤 ★★★
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list():
    """取得台股清單 (排除金融/ETF)"""
    try:
        # ★★★ 關鍵修正：強制更新代碼，避免抓不到新股 ★★★
        twstock.__update_codes()
        
        tse = twstock.twse
        otc = twstock.tpex
        stock_dict = {}
        exclude_industries = ['金融保險業', '存託憑證']
        
        for code, info in tse.items():
            if info.type == '股票' and info.group not in exclude_industries:
                stock_dict[f"{code}.TW"] = {'name': info.name, 'code': code, 'group': info.group}
        for code, info in otc.items():
            if info.type == '股票' and info.group not in exclude_industries:
                stock_dict[f"{code}.TWO"] = {'name': info.name, 'code': code, 'group': info.group}
        return stock_dict
    except Exception as e:
        print(f"Error fetching stock list: {e}")
        return {}

def calculate_kd_values(df, n=9):
    try:
        low_min = df['Low'].rolling(window=n).min()
        high_max = df['High'].rolling(window=n).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        k, d = 50, 50
        for r in rsv:
            k = (2/3) * k + (1/3) * r
            d = (2/3) * d + (1/3) * k
        return k, d
    except:
        return 50, 50

# --- 策略回測核心函數 ---
def run_strategy_backtest(stock_dict, progress_bar, use_trend_up, use_treasure, use_vol, use_royal, min_vol_threshold):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    OBSERVE_DAYS = 20 if use_royal else 10
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if not data.empty:
                # 處理 MultiIndex (yfinance 新版相容性)
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        df_c = data.xs('Close', axis=1, level=0)
                        df_v = data.xs('Volume', axis=1, level=0)
                        df_l = data.xs('Low', axis=1, level=0)
                        df_h = data.xs('High', axis=1, level=0)
                    except:
                        continue
                else:
                    df_c = data['Close']
                    df_v = data['Volume']
                    df_l = data['Low']
                    df_h = data['High']

                ma200_df = df_c.rolling(window=200).mean()
                if use_royal:
                    ma20_df = df_c.rolling(window=20).mean()
                    ma60_df = df_c.rolling(window=60).mean()
                
                scan_window = df_c.index[-90:] 
                
                for ticker in df_c.columns:
                    try:
                        c_series = df_c[ticker]
                        v_series = df_v[ticker]
                        l_series = df_l[ticker]
                        h_series = df_h[ticker]
                        ma200_series = ma200_df[ticker]
                        
                        if use_royal:
                            ma20_series = ma20_df[ticker]
                            ma60_series = ma60_df[ticker]
                        
                        stock_name = stock_dict.get(ticker, {}).get('name', ticker)
                        total_len = len(c_series)

                        for date in scan_window:
                            if pd.isna(ma200_series.get(date)): continue
                            if date not in c_series.index: continue

                            idx = c_series.index.get_loc(date)
                            if idx < 200: continue 

                            close_p = c_series.iloc[idx]
                            vol = v_series.iloc[idx]
                            prev_vol = v_series.iloc[idx-1]
                            ma200_val = ma200_series.iloc[idx]
                            
                            if vol < (min_vol_threshold * 1000): continue
                            if ma200_val == 0 or prev_vol == 0: continue

                            is_match = False
                            
                            if use_royal:
                                ma20_val = ma20_series.iloc[idx]
                                ma60_val = ma60_series.iloc[idx]
                                if (close_p > ma20_val) and (ma20_val > ma60_val) and (ma60_val > ma200_val):
                                    is_match = True
                            else:
                                low_p = l_series.iloc[idx]
                                ma_val_20ago = ma200_series.iloc[idx-20]
                                
                                if use_trend_up and (ma200_val <= ma_val_20ago): continue
                                if use_vol and (vol <= prev_vol * 1.5): continue

                                if use_treasure:
                                    start_idx = idx - 7
                                    if start_idx < 0: continue
                                    recent_c = c_series.iloc[start_idx : idx+1]
                                    recent_ma = ma200_series.iloc[start_idx : idx+1]
                                    cond_today_up = recent_c.iloc[-1] > recent_ma.iloc[-1]
                                    past_c = recent_c.iloc[:-1]
                                    past_ma = recent_ma.iloc[:-1]
                                    cond_past_down = (past_c < past_ma).any()
                                    if cond_today_up and cond_past_down: is_match = True
                                else:
                                    cond_near = (low_p <= ma200_val * 1.03) and (low_p >= ma200_val * 0.90) 
                                    cond_up = (close_p > ma200_val)
                                    if cond_near and cond_up: is_match = True
                            
                            if is_match:
                                month_str = date.strftime('%m月')
                                days_after_signal = total_len - 1 - idx
                                
                                final_profit_pct = 0.0
                                result_status = "觀察中"
                                is_watching = False

                                if days_after_signal < 1: 
                                    is_watching = True
                                    final_profit_pct = 0.0
                                    
                                elif use_royal:
                                    is_watching = True 
                                    current_price = c_series.iloc[-1]
                                    final_profit_pct = (current_price - close_p) / close_p * 100
                                    check_days = min(days_after_signal, OBSERVE_DAYS)
                                    for d in range(1, check_days + 1):
                                        day_idx = idx + d
                                        day_high = h_series.iloc[day_idx]
                                        day_close = c_series.iloc[day_idx]
                                        day_ma200 = ma200_series.iloc[day_idx]
                                        
                                        if day_high >= close_p * 1.10:
                                            final_profit_pct = 10.0
                                            result_status = "Win (止盈) 👑"
                                            is_watching = False 
                                            break
                                        if day_close < day_ma200:
                                            final_profit_pct = (day_close - close_p) / close_p * 100
                                            result_status = "Loss (停損) 🛑"
                                            is_watching = False 
                                            break
                                    if is_watching and days_after_signal >= OBSERVE_DAYS:
                                        end_close = c_series.iloc[idx + OBSERVE_DAYS]
                                        final_profit_pct = (end_close - close_p) / close_p * 100
                                        if final_profit_pct > 0: result_status = "Win (期滿)"
                                        else: result_status = "Loss (期滿)"
                                        is_watching = False

                                else:
                                    if days_after_signal < OBSERVE_DAYS:
                                        current_price = c_series.iloc[-1]
                                        final_profit_pct = (current_price - close_p) / close_p * 100
                                        is_watching = True
                                    else:
                                        future_highs = h_series.iloc[idx+1 : idx+1+OBSERVE_DAYS]
                                        max_price = future_highs.max()
                                        final_profit_pct = (max_price - close_p) / close_p * 100
                                        if final_profit_pct > 3.0: result_status = "Win 🏆"
                                        elif final_profit_pct > 0: result_status = "Win ↗️"
                                        else: result_status = "Loss ↘️"

                                results.append({
                                    '月份': '👀 關注中' if is_watching else month_str,
                                    '股號': ticker.replace(".TW", "").replace(".TWO", ""), # 縮短欄位
                                    '股名': stock_name, # 縮短欄位
                                    '代號與名稱': f"{ticker.replace('.TW', '').replace('.TWO', '')} {stock_name}", # 手機專用合併欄位
                                    '訊號日期': date.strftime('%Y-%m-%d'),
                                    '訊號價': float(close_p),
                                    '最高漲幅': float(final_profit_pct),
                                    '結果': "👀 觀察中" if is_watching else result_status
                                })
                                break 
                    except:
                        continue
        except:
            pass
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中...({int(progress*100)}%)")
        
    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar, status_text):
    if not stock_dict: return pd.DataFrame()
    
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 30
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data_list = []

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="1y", interval="1d", progress=False, auto_adjust=False)
            if not data.empty:
                # 處理 MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        df_c = data.xs('Close', axis=1, level=0)
                        df_h = data.xs('High', axis=1, level=0)
                        df_l = data.xs('Low', axis=1, level=0)
                        df_v = data.xs('Volume', axis=1, level=0)
                    except: continue
                else:
                    df_c = data['Close']
                    df_h = data['High']
                    df_l = data['Low']
                    df_v = data['Volume']

                ma200_df = df_c.rolling(window=200).mean()
                ma20_df = df_c.rolling(window=20).mean()
                ma60_df = df_c.rolling(window=60).mean()

                last_price_series = df_c.iloc[-1]
                last_ma200_series = ma200_df.iloc[-1]
                last_ma20_series = ma20_df.iloc[-1]
                last_ma60_series = ma60_df.iloc[-1]
                prev_ma200_series = ma200_df.iloc[-21] 
                
                last
