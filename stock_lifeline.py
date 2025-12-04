import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import os
import matplotlib.pyplot as plt # 確保表格顏色正常

# --- 1. 網頁設定 ---
VER = "ver5.0_CrownStrategy"
st.set_page_config(page_title=f"🍍 旺來-台股生命線({VER})", layout="wide")

# --- 2. 核心功能區 ---
@st.cache_data(ttl=3600)
def get_stock_list():
    """取得台股清單 (排除金融/ETF)"""
    try:
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
    except:
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

def calculate_obv(df):
    try:
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv
    except:
        return pd.Series(0, index=df.index)

# --- 策略最佳化擂台函數 (含動態出場邏輯) ---
def run_optimization_tournament(stock_dict, progress_bar):
    raw_signals = [] 
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex): pass 
            
            if not data.empty:
                try:
                    df_c = data['Close']
                    df_v = data['Volume']
                    df_l = data['Low']
                    df_h = data['High']
                except KeyError: continue
                
                if isinstance(df_c, pd.Series):
                    ticker = batch[0]
                    df_c = df_c.to_frame(name=ticker)
                    df_v = df_v.to_frame(name=ticker)
                    df_l = df_l.to_frame(name=ticker)
                    df_h = df_h.to_frame(name=ticker)

                # 計算均線
                ma20_df = df_c.rolling(window=20).mean()
                ma60_df = df_c.rolling(window=60).mean()
                ma200_df = df_c.rolling(window=200).mean()
                
                # 計算全體 OBV
                obv_df = pd.DataFrame(index=df_c.index, columns=df_c.columns)
                for col in df_c.columns:
                    obv_df[col] = calculate_obv(pd.DataFrame({'Close': df_c[col], 'Volume': df_v[col]}))

                scan_window_idx = df_c.index[-250:-25] # 預留25天給動態出場
                
                for ticker in df_c.columns:
                    try:
                        c_series = df_c[ticker]
                        v_series = df_v[ticker]
                        l_series = df_l[ticker]
                        h_series = df_h[ticker]
                        ma200_series = ma200_df[ticker]
                        ma20_series = ma20_df[ticker]
                        ma60_series = ma60_df[ticker]
                        obv_series = obv_df[ticker]
                        
                        if c_series.isna().sum() > 100 or ma200_series.isna().all(): continue

                        for date in scan_window_idx:
                            if pd.isna(ma200_series[date]): continue
                            idx = c_series.index.get_loc(date)
                            if idx < 60: continue 

                            close_p = float(c_series.iloc[idx])
                            low_p = float(l_series.iloc[idx])
                            vol = float(v_series.iloc[idx])
                            prev_vol = float(v_series.iloc[idx-1])
                            ma200_val = float(ma200_series.iloc[idx])
                            ma20_val = float(ma20_series.iloc[idx])
                            ma60_val = float(ma60_series.iloc[idx])
                            ma200_20ago = float(ma200_series.iloc[idx-20])
                            
                            if ma200_val == 0 or prev_vol == 0: continue

                            # --- 訊號判斷 ---
                            cond_near = (low_p <= ma200_val * 1.03) and (low_p >= ma200_val * 0.90)
                            cond_up = (close_p > ma200_val)
                            is_basic_signal = cond_near and cond_up # 基礎訊號
                            
                            tag_trend_up = (ma200_val > ma200_20ago)
                            tag_vol_double = (vol > prev_vol * 1.5)
                            
                            obv_now = obv_series.iloc[idx]
                            obv_week_ago = obv_series.iloc[idx-5]
                            tag_obv_in = obv_now > obv_week_ago

                            # 皇冠特選：多頭排列 (價格 > 20 > 60 > 200)
                            tag_crown = (close_p > ma20_val) and (ma20_val > ma60_val) and (ma60_val > ma200_val) and tag_trend_up

                            # 浴火重生
                            tag_treasure = False
                            start_idx = idx - 7
                            if start_idx >= 0:
                                recent_c = c_series.iloc[start_idx : idx+1]
                                recent_ma = ma200_series.iloc[start_idx : idx+1]
                                cond_today_up = recent_c.iloc[-1] > recent_ma.iloc[-1]
                                cond_past_down = (recent_c.iloc[:-1] < recent_ma.iloc[:-1]).any()
                                if cond_today_up and cond_past_down:
                                    tag_treasure = True

                            if not is_basic_signal and not tag_treasure and not tag_crown:
                                continue
                                
                            # --- 績效計算 (分為 一般持有20天 vs 動態出場) ---
                            if idx + 20 < len(c_series):
                                # 1. 傳統：持有20天
                                exit_price_static = float(c_series.iloc[idx + 20])
                                profit_static = (exit_price_static - close_p) / close_p * 100
                                is_win_static = profit_static > 0

                                # 2. 動態：停利(+10%) 或 停損(跌破MA200)
                                exit_price_dynamic = float(c_series.iloc[idx + 20]) # 預設
                                status_dynamic = "Hold"
                                
                                # 逐日掃描未來 20 天
                                for future_i in range(1, 21):
                                    f_idx = idx + future_i
                                    if f_idx >= len(c_series): break
                                    
                                    f_high = float(h_series.iloc[f_idx])
                                    f_close = float(c_series.iloc[f_idx])
                                    f_ma200 = float(ma200_series.iloc[f_idx])
                                    
                                    # 停利：最高價碰到 +10%
                                    if f_high >= close_p * 1.10:
                                        exit_price_dynamic = close_p * 1.10
                                        status_dynamic = "TakeProfit"
                                        break
                                    
                                    # 停損：收盤跌破 MA200 (容忍度 99%)
                                    if f_close < f_ma200 * 0.99:
                                        exit_price_dynamic = f_close
                                        status_dynamic = "StopLoss"
                                        break
                                
                                profit_dynamic = (exit_price_dynamic - close_p) / close_p * 100
                                is_win_dynamic = profit_dynamic > 0
                                
                            else:
                                continue 

                            raw_signals.append({
                                'Profit_Static': profit_static,
                                'Is_Win_Static': is_win_static,
                                'Profit_Dynamic': profit_dynamic,
                                'Is_Win_Dynamic': is_win_dynamic,
                                'Tag_Trend_Up': tag_trend_up,
                                'Tag_Vol_Double': tag_vol_double,
                                'Tag_Treasure': tag_treasure,
                                'Tag_OBV_In': tag_obv_in,
                                'Tag_Crown': tag_crown,
                                'Is_Basic_Near': is_basic_signal
                            })

                    except Exception: continue
        except: pass
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"全策略掃描中...({int(progress*100)}%)")
        
    return pd.DataFrame(raw_signals)

# --- 單一回測函數 (支援皇冠策略) ---
def run_strategy_backtest(stock_dict, progress_bar, use_trend_up, use_treasure, use_vol, use_obv, use_crown):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex): pass
            if not data.empty:
                try:
                    df_c = data['Close']
                    df_v = data['Volume']
                    df_l = data['Low']
                    df_h = data['High']
                except KeyError: continue
                
                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])

                ma20_df = df_c.rolling(window=20).mean()
                ma60_df = df_c.rolling(window=60).mean()
                ma200_df = df_c.rolling(window=200).mean()
                
                obv_df = pd.DataFrame(index=df_c.index, columns=df_c.columns)
                for col in df_c.columns:
                    obv_df[col] = calculate_obv(pd.DataFrame({'Close': df_c[col], 'Volume': df_v[col]}))

                scan_window = df_c.index[-250:-25] 
                
                for ticker in df_c.columns:
                    try:
                        c_series = df_c[ticker]
                        v_series = df_v[ticker]
                        l_series = df_l[ticker]
                        h_series = df_h[ticker]
                        ma200_series = ma200_df[ticker]
                        ma20_series = ma20_df[ticker]
                        ma60_series = ma60_df[ticker]
                        obv_series = obv_df[ticker]
                        
                        stock_name = stock_dict.get(ticker, {}).get('name', ticker)
                        stock_code = stock_dict.get(ticker, {}).get('code', ticker.split('.')[0])
                        
                        for date in scan_window:
                            if pd.isna(ma200_series[date]): continue
                            idx = c_series.index.get_loc(date)
                            if idx < 60: continue 

                            close_p = float(c_series.iloc[idx])
                            low_p = float(l_series.iloc[idx])
                            vol = float(v_series.iloc[idx])
                            prev_vol = float(v_series.iloc[idx-1])
                            ma200_val = float(ma200_series.iloc[idx])
                            
                            # OBV Check
                            obv_now = obv_series.iloc[idx]
                            obv_week_ago = obv_series.iloc[idx-5]
                            is_obv_up = obv_now > obv_week_ago

                            if ma200_val == 0 or prev_vol == 0: continue
                            is_match = False
                            
                            # 條件判斷
                            if use_crown:
                                # 皇冠策略：多頭排列 + 趨勢向上
                                ma20 = float(ma20_series.iloc[idx])
                                ma60 = float(ma60_series.iloc[idx])
                                ma200_20ago = float(ma200_series.iloc[idx-20])
                                is_trend_up = ma200_val > ma200_20ago
                                is_perfect_order = (close_p > ma20) and (ma20 > ma60) and (ma60 > ma200_val)
                                if is_trend_up and is_perfect_order:
                                    is_match = True
                            else:
                                # 一般策略
                                if use_trend_up and (ma200_val <= float(ma200_series.iloc[idx-20])): continue
                                if use_vol and (vol <= prev_vol * 1.5): continue
                                if use_obv and not is_obv_up: continue

                                if use_treasure:
                                    start_idx = idx - 7
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
                                # 動態出場回測
                                if idx + 20 < len(c_series):
                                    exit_price = float(c_series.iloc[idx + 20])
                                    status = "持有20天"
                                    
                                    # 如果是皇冠策略，強制使用動態出場
                                    if use_crown:
                                        for future_i in range(1, 21):
                                            f_idx = idx + future_i
                                            if f_idx >= len(c_series): break
                                            f_h = float(h_series.iloc[f_idx])
                                            f_c = float(c_series.iloc[f_idx])
                                            f_ma = float(ma200_series.iloc[f_idx])
                                            
                                            if f_h >= close_p * 1.10:
                                                exit_price = close_p * 1.10
                                                status = "🎯 停利 (+10%)"
                                                break
                                            if f_c < f_ma * 0.99:
                                                exit_price = f_c
                                                status = "🛡️ 停損 (破線)"
                                                break
                                    
                                    profit_pct = (exit_price - close_p) / close_p * 100
                                    results.append({
                                        'StockID': stock_code,
                                        '名稱': stock_name,
                                        'Date': date,
                                        '訊號日期': date.strftime('%Y-%m-%d'),
                                        '訊號價': round(float(close_p), 2),
                                        '出場價': round(float(exit_price), 2),
                                        '報酬率(%)': round(float(profit_pct), 2),
                                        '結果': status
                                    })
                    except Exception: continue
        except: pass
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中...({int(progress*100)}%)")
        
    return pd.DataFrame(results) if results else pd.DataFrame()
    # --- 即時資料抓取 ---
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
            if isinstance(data.columns, pd.MultiIndex): pass
            
            if not data.empty:
                try:
                    df_c = data['Close']
                    df_h = data['High']
                    df_l = data['Low']
                    df_v = data['Volume']
                except KeyError: continue

                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])

                ma20_df = df_c.rolling(window=20).mean()
                ma60_df = df_c.rolling(window=60).mean()
                ma200_df = df_c.rolling(window=200).mean()
                obv_df = pd.DataFrame(index=df_c.index, columns=df_c.columns)
                for col in df_c.columns:
                    obv_df[col] = calculate_obv(pd.DataFrame({'Close': df_c[col], 'Volume': df_v[col]}))
                
                last_idx = -1
                
                for ticker in df_c.columns:
                    try:
                        price = float(df_c[ticker].iloc[last_idx])
                        ma200 = float(ma200_df[ticker].iloc[last_idx])
                        ma20 = float(ma20_df[ticker].iloc[last_idx])
                        ma60 = float(ma60_df[ticker].iloc[last_idx])
                        vol = float(df_v[ticker].iloc[last_idx])
                        prev_vol = float(df_v[ticker].iloc[last_idx-1])
                        
                        obv_now = obv_df[ticker].iloc[last_idx]
                        obv_prev = obv_df[ticker].iloc[last_idx-6]
                        is_obv_in = obv_now > obv_prev
                        
                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0: continue

                        ma_trend = "⬆️向上" if ma200 >= ma200_df[ticker].iloc[last_idx-20] else "⬇️向下"
                        
                        is_crown = (price > ma20) and (ma20 > ma60) and (ma60 > ma200) and (ma200 > ma200_df[ticker].iloc[last_idx-20])

                        is_treasure = False
                        recent_c = df_c[ticker].iloc[-8:]
                        recent_ma = ma200_df[ticker].iloc[-8:]
                        if len(recent_c) >= 8:
                            cond_today_up = float(recent_c.iloc[-1]) > float(recent_ma.iloc[-1])
                            cond_past_down = (recent_c.iloc[:-1] < recent_ma.iloc[:-1]).any()
                            if cond_today_up and cond_past_down: is_treasure = True

                        stock_df = pd.DataFrame({'Close': df_c[ticker], 'High': df_h[ticker], 'Low': df_l[ticker]}).dropna()
                        k_val, d_val = 0, 0
                        if len(stock_df) >= 9:
                            k_val, d_val = calculate_kd_values(stock_df)

                        bias = ((price - ma200) / ma200) * 100
                        stock_info = stock_dict.get(ticker)
                        if not stock_info: continue

                        raw_data_list.append({
                            '代號': stock_info['code'],
                            '名稱': stock_info['name'],
                            '完整代號': ticker,
                            '收盤價': round(price, 2),
                            '生命線': round(ma200, 2),
                            '生命線趨勢': ma_trend,
                            '乖離率(%)': round(bias, 2),
                            'abs_bias': abs(bias),
                            '成交量': int(vol),
                            '昨日成交量': int(prev_vol),
                            'K值': round(float(k_val), 2),
                            'D值': round(float(d_val), 2),
                            '位置': "🟢生命線上" if price >= ma200 else "🔴生命線下",
                            '浴火重生': is_treasure,
                            'OBV趨勢': "🔥吸籌" if is_obv_in else "☁️一般",
                            '皇冠型態': is_crown
                        })
                    except Exception: continue
        except: pass
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"挖掘寶藏中...({int(current_progress*100)}%)")
        time.sleep(0.02)
    return pd.DataFrame(raw_data_list)

def plot_stock_chart(ticker, name):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df.dropna()
        if df.empty: return
        df['200MA'] = df['Close'].rolling(window=200).mean()
        df['60MA'] = df['Close'].rolling(window=60).mean()
        df['20MA'] = df['Close'].rolling(window=20).mean()
        plot_df = df.tail(120).copy()
        plot_df['DateStr'] = plot_df.index.strftime('%Y-%m-%d')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['Close'], mode='lines', name='收盤價', line=dict(color='#00CC96', width=2.5)))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['20MA'], mode='lines', name='月線(20MA)', line=dict(color='#AB63FA', width=1)))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['60MA'], mode='lines', name='季線(60MA)', line=dict(color='#19D3F3', width=1)))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['200MA'], mode='lines', name='生命線(200MA)', line=dict(color='#FFA15A', width=3)))
        
        fig.update_layout(title=f"📊 {name} ({ticker})", yaxis_title='價格', height=500, hovermode="x unified", legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    except: st.error("繪圖失敗")

# --- 3. 介面顯示區 ---
st.title(f"🍍 {VER} 旺來-台股生命線")
st.markdown("---")

if 'master_df' not in st.session_state: st.session_state['master_df'] = None
if 'last_update' not in st.session_state: st.session_state['last_update'] = None
if 'backtest_result' not in st.session_state: st.session_state['backtest_result'] = None
if 'optimizer_result' not in st.session_state: st.session_state['optimizer_result'] = None

with st.sidebar:
    st.header("資料庫管理")
    if st.button("🚨 強制重置系統"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun
