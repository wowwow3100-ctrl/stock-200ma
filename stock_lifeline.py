import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import plotly.express as px
import os
import uuid
import csv

# --- 1. 網頁設定 ---
VER = "ver5.1 (Logic Synced)"
st.set_page_config(page_title=f"🍍 旺來-台股生命線({VER})", layout="wide")

# --- 時間校正工具 (UTC+8) ---
def get_taiwan_time():
    """取得台灣時間 (UTC+8)"""
    utc_now = datetime.now(timezone.utc)
    tw_time = utc_now + timedelta(hours=8)
    return tw_time

def get_taiwan_time_str():
    return get_taiwan_time().strftime("%Y-%m-%d %H:%M:%S")

# --- 流量紀錄與後台功能 ---
LOG_FILE = "traffic_log.csv"

def get_remote_ip():
    """取得使用者 IP"""
    try:
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            headers = st.context.headers
            if headers and "X-Forwarded-For" in headers:
                return headers["X-Forwarded-For"].split(",")[0]
        
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers and "X-Forwarded-For" in headers:
            return headers["X-Forwarded-For"].split(",")[0]
    except:
        pass
    return "Unknown/Local"

def log_traffic():
    """紀錄使用者訪問"""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())[:8] 
        st.session_state['has_logged'] = False

    if not st.session_state['has_logged']:
        current_time = get_taiwan_time_str()
        user_ip = get_remote_ip()
        session_id = st.session_state['session_id']
        
        file_exists = os.path.exists(LOG_FILE)
        try:
            with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["時間", "IP位址", "Session_ID", "頁面動作"])
                writer.writerow([current_time, user_ip, session_id, "進入首頁"])
        except:
            pass 
        st.session_state['has_logged'] = True

log_traffic()

# --- 2. 核心功能區 ---

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list():
    """取得台股清單 (排除金融/ETF)"""
    try:
        tse = twstock.twse
        otc = twstock.tpex
        stock_dict = {}
        exclude_industries = ['金融保險業', '存託憑證']
        
        # 上市
        for code, info in tse.items():
            if info.type == '股票':
                if info.group not in exclude_industries:
                    stock_dict[f"{code}.TW"] = {'name': info.name, 'code': code, 'group': info.group}
        
        # 上櫃
        for code, info in otc.items():
            if info.type == '股票':
                if info.group not in exclude_industries:
                    key = f"{code}.TWO"
                    if f"{code}.TW" not in stock_dict: 
                        stock_dict[key] = {'name': info.name, 'code': code, 'group': info.group}
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
        
        k_list = []
        d_list = []
        
        for r in rsv:
            k = (2/3) * k + (1/3) * r
            d = (2/3) * d + (1/3) * k
            k_list.append(k)
            d_list.append(d)
        
        if not k_list: return 50, 50
        return k_list[-1], d_list[-1]
    except:
        return 50, 50

# --- 更新功能：週報掃描 (已修正邏輯同步問題) ---
def scan_period_signals(stock_dict, days_lookback, progress_bar, min_vol, bias_thresh, strategy_type, 
                        use_trend_up, use_trend_down, use_kd, use_vol_double, use_burst_vol):
    """
    掃描過去 N 天內符合條件的股票 (同步所有側邊欄條件)
    """
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    # 內部小工具：計算連續站穩天數
    def calculate_streak(ma_series, close_series, start_idx):
        streak = 0
        for k in range(start_idx + 1, len(close_series)):
            if close_series.iloc[k] > ma_series.iloc[k]:
                streak += 1
            else:
                streak = 0 
        return streak

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            # 必須抓長一點的資料以計算 KD/MA
            data = yf.download(batch, period="9mo", interval="1d", progress=False, auto_adjust=False)
            if data.empty: continue
            
            try:
                df_c = data['Close']
                df_v = data['Volume']
                df_l = data['Low']
                df_h = data['High']
                df_o = data['Open']
            except KeyError: continue

            if isinstance(df_c, pd.Series):
                df_c = df_c.to_frame(name=batch[0])
                df_v = df_v.to_frame(name=batch[0])
                df_l = df_l.to_frame(name=batch[0])
                df_h = df_h.to_frame(name=batch[0])
                df_o = df_o.to_frame(name=batch[0])

            ma200_df = df_c.rolling(window=200).mean()
            vol_ma5_df = df_v.rolling(window=5).mean()
            
            for ticker in df_c.columns:
                try:
                    c_series = df_c[ticker].dropna()
                    if len(c_series) < 200: continue
                    
                    # 準備所有需要的 Series
                    ma200_series = ma200_df[ticker]
                    v_series = df_v[ticker]
                    l_series = df_l[ticker]
                    h_series = df_h[ticker]
                    o_series = df_o[ticker]
                    vol_ma5_series = vol_ma5_df[ticker]
                    
                    stock_info = stock_dict.get(ticker, {})
                    name = stock_info.get('name', ticker)
                    industry = stock_info.get('group', '')
                    
                    current_price = c_series.iloc[-1]
                    start_scan_idx = len(c_series) - 1 
                    
                    # 倒帶掃描
                    for lookback in range(days_lookback):
                        day_idx = start_scan_idx - lookback 
                        if day_idx < 200: break
                        
                        date = c_series.index[day_idx]
                        close_p = c_series.iloc[day_idx]
                        ma200_val = ma200_series.iloc[day_idx]
                        vol = v_series.iloc[day_idx]
                        prev_vol = v_series.iloc[day_idx-1] if day_idx > 0 else 0
                        vol_ma5_val = vol_ma5_series.iloc[day_idx-1] if day_idx > 0 else 0
                        
                        # --- 1. 基礎與成交量過濾 ---
                        if vol < (min_vol * 1000) or pd.isna(ma200_val) or ma200_val == 0: continue
                        
                        is_signal = False
                        
                        # --- 2. 策略核心判斷 (與今日篩選同步) ---
                        
                        # A. 生命線趨勢過濾
                        ma_trend_ok = True
                        if use_trend_up:
                            # 簡單判斷：今日 MA > 20日前 MA (模擬月趨勢)
                            ma_prev = ma200_series.iloc[day_idx-20] if day_idx >= 20 else 0
                            if ma200_val <= ma_prev: ma_trend_ok = False
                        elif use_trend_down:
                            ma_prev = ma200_series.iloc[day_idx-20] if day_idx >= 20 else 0
                            if ma200_val >= ma_prev: ma_trend_ok = False
                        
                        if not ma_trend_ok: continue

                        # B. 成交量倍增過濾
                        if use_vol_double and (vol <= prev_vol * 1.5): continue

                        # C. 爆量起漲過濾
                        if use_burst_vol:
                            open_p = o_series.iloc[day_idx]
                            # 量 > 5日均量1.5倍 且 收紅K
                            if (vol <= vol_ma5_val * 1.5) or (close_p <= open_p): continue

                        # D. KD 黃金交叉過濾
                        if use_kd:
                            # 擷取當天之前的資料來算 KD (避免未來數據)
                            # 為了效率，只取最近 20 天算 KD
                            sub_start = max(0, day_idx - 30)
                            sub_df = pd.DataFrame({
                                'Close': c_series.iloc[sub_start:day_idx+1],
                                'High': h_series.iloc[sub_start:day_idx+1],
                                'Low': l_series.iloc[sub_start:day_idx+1]
                            })
                            k_val, d_val = calculate_kd_values(sub_df)
                            if not (k_val > d_val): continue

                        # E. 主策略 (守護 vs 浴火)
                        if strategy_type == "🛡️ 守護生命線 (反彈/支撐)":
                            bias = (close_p - ma200_val) / ma200_val * 100
                            # 條件：站上生命線 且 乖離率在範圍內
                            if close_p > ma200_val and 0 < bias <= bias_thresh:
                                is_signal = True
                                
                        elif strategy_type == "🔥 浴火重生 (假跌破)":
                            start_check = day_idx - 7
                            if start_check >= 0:
                                subset_c = c_series.iloc[start_check : day_idx+1]
                                subset_ma = ma200_series.iloc[start_check : day_idx+1]
                                if len(subset_c) >= 8:
                                    cond_today_up = subset_c.iloc[-1] > subset_ma.iloc[-1]
                                    past_c = subset_c.iloc[:-1]
                                    past_ma = subset_ma.iloc[:-1]
                                    cond_past_down = (past_c < past_ma).any()
                                    if cond_today_up and cond_past_down:
                                        is_signal = True

                        if is_signal:
                            profit_pct = (current_price - close_p) / close_p * 100
                            streak_days = calculate_streak(ma200_series, c_series, day_idx)
                            
                            status = "🟢 獲利" if profit_pct > 0 else "🔴 虧損"
                            if current_price < ma200_series.iloc[-1]:
                                status = "💀 跌破"

                            results.append({
                                '訊號日期': date.strftime('%Y-%m-%d'),
                                '距今': f"{lookback}天",
                                '代號': ticker.replace(".TW", "").replace(".TWO", ""),
                                '名稱': name,
                                '產業': industry,
                                '訊號價': round(close_p, 2),
                                '現價': round(current_price, 2),
                                '至今漲跌(%)': round(profit_pct, 2),
                                '成交量': int(vol),
                                '站穩': streak_days,
                                '狀態': status
                            })
                            # 每個股票每天只抓一次，但我們希望看到連續幾天的訊號，所以這裡不 break?
                            # 為了週報清晰，如果同一檔股票連續3天都符合，顯示3筆是合理的(堆疊)
                            pass 
                except: continue
        except: 
            time.sleep(0.1) 
            continue
        
        time.sleep(0.1) 
        prog = (i + 1) / total_batches
        progress_bar.progress(prog, text=f"正在編制本週戰報 (同步篩選邏輯)...({int(prog*100)}%)")

    return pd.DataFrame(results)

# --- 策略回測核心函數 ---
def run_strategy_backtest(stock_dict, progress_bar, use_trend_up, use_treasure, use_vol, min_vol_threshold, use_burst_vol):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    OBSERVE_DAYS = 10 
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if data is None or data.empty: continue

            try:
                df_c = data['Close']
                df_v = data['Volume']
                df_l = data['Low']
                df_h = data['High']
                df_o = data['Open'] 
            except KeyError: continue
            
            if isinstance(df_c, pd.Series):
                df_c = df_c.to_frame(name=batch[0])
                df_v = df_v.to_frame(name=batch[0])
                df_l = df_l.to_frame(name=batch[0])
                df_h = df_h.to_frame(name=batch[0])
                df_o = df_o.to_frame(name=batch[0])

            ma200_df = df_c.rolling(window=200).mean()
            vol_ma5_df = df_v.rolling(window=5).mean()
            scan_window = df_c.index[-120:]
            
            for ticker in df_c.columns:
                try:
                    c_series = df_c[ticker]
                    v_series = df_v[ticker]
                    l_series = df_l[ticker]
                    h_series = df_h[ticker]
                    o_series = df_o[ticker]
                    ma200_series = ma200_df[ticker]
                    vol_ma5_series = vol_ma5_df[ticker]
                    
                    stock_info = stock_dict.get(ticker, {})
                    stock_name = stock_info.get('name', ticker)
                    stock_industry = stock_info.get('group', '其他')

                    total_len = len(c_series)

                    for date in scan_window:
                        if pd.isna(ma200_series.get(date)): continue
                        if date not in c_series.index: continue

                        idx = c_series.index.get_loc(date)
                        if idx < 200: continue 

                        close_p = c_series.iloc[idx]
                        open_p = o_series.iloc[idx]
                        vol = v_series.iloc[idx]
                        prev_vol = v_series.iloc[idx-1]
                        ma200_val = ma200_series.iloc[idx]
                        vol_ma5_val = vol_ma5_series.iloc[idx-1] 
                        
                        if vol < (min_vol_threshold * 1000): continue
                        if ma200_val == 0 or prev_vol == 0: continue

                        is_match = False
                        low_p = l_series.iloc[idx]
                        ma_val_20ago = ma200_series.iloc[idx-20]
                        
                        if use_trend_up and (ma200_val <= ma_val_20ago): continue
                        if use_vol and (vol <= prev_vol * 1.5): continue

                        if use_burst_vol:
                            if vol <= (vol_ma5_val * 1.5) or close_p <= open_p: continue

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
                            else:
                                if days_after_signal < OBSERVE_DAYS:
                                    current_price = c_series.iloc[-1]
                                    final_profit_pct = (current_price - close_p) / close_p * 100
                                    is_watching = True
                                else:
                                    future_highs = h_series.iloc[idx+1 : idx+1+OBSERVE_DAYS]
                                    max_price = future_highs.max()
                                    final_profit_pct = (max_price - close_p) / close_p * 100
                                    if final_profit_pct > 3.0: result_status = "驗證成功 🏆"
                                    elif final_profit_pct > 0: result_status = "Win (反彈)"
                                    else: result_status = "Loss 📉"

                            results.append({
                                '訊號日期': date,
                                '月份': '👀 關注中' if is_watching else month_str,
                                '代號': ticker.replace(".TW", "").replace(".TWO", ""),
                                '名稱': stock_name,
                                '產業': stock_industry,
                                '訊號價': round(close_p, 2),
                                '最高漲幅(%)': round(final_profit_pct, 2),
                                '結果': "觀察中" if is_watching else result_status,
                                'is_win': 1 if final_profit_pct > 0 else 0
                            })
                            break 
                except: continue
        except Exception:
            time.sleep(1) 
            continue
        
        time.sleep(0.1) 
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中 (計算分月數據)...({int(progress*100)}%)")
        
    if not results:
        return pd.DataFrame(columns=['訊號日期', '月份', '代號', '名稱', '產業', '訊號價', '最高漲幅(%)', '結果', 'is_win'])

    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar, status_text):
    if not stock_dict: return pd.DataFrame()
    
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data_list = []

    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="1y", interval="1d", progress=False, auto_adjust=False)
            if not data.empty:
                try:
                    df_c = data['Close']
                    df_h = data['High']
                    df_l = data['Low']
                    df_o = data['Open']
                    df_v = data['Volume']
                except KeyError: continue

                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_o = df_o.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])

                ma200_df = df_c.rolling(window=200).mean()
                ma20_df = df_c.rolling(window=20).mean()
                ma60_df = df_c.rolling(window=60).mean()
                vol_ma5_df = df_v.rolling(window=5).mean()

                last_price_series = df_c.iloc[-1]
                last_open_series = df_o.iloc[-1]
                last_ma200_series = ma200_df.iloc[-1]
                last_ma20_series = ma20_df.iloc[-1]
                last_ma60_series = ma60_df.iloc[-1]
                prev_ma200_series = ma200_df.iloc[-21] 
                
                last_vol_series = df_v.iloc[-1]
                prev_vol_series = df_v.iloc[-2]
                last_vol_ma5_series = vol_ma5_df.iloc[-2]

                recent_close_df = df_c.iloc[-8:]
                recent_ma200_df = ma200_df.iloc[-8:]

                for ticker in df_c.columns:
                    try:
                        price = last_price_series[ticker]
                        open_p = last_open_series[ticker]
                        ma200 = last_ma200_series[ticker]
                        ma20 = last_ma20_series[ticker]
                        ma60 = last_ma60_series[ticker]
                        prev_ma200 = prev_ma200_series[ticker]
                        
                        vol = last_vol_series[ticker]
                        prev_vol = prev_vol_series[ticker]
                        vol_ma5 = last_vol_ma5_series[ticker]
                        
                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0: continue

                        ma_trend = "⬆️向上" if ma200 >= prev_ma200 else "⬇️向下"

                        is_treasure = False
                        my_recent_c = recent_close_df[ticker]
                        my_recent_ma = recent_ma200_df[ticker]
                        if len(my_recent_c) >= 8:
                            cond_today_up = my_recent_c.iloc[-1] > my_recent_ma.iloc[-1]
                            past_c = my_recent_c.iloc[:-1]
                            past_ma = my_recent_ma.iloc[:-1]
                            cond_past_down = (past_c < past_ma).any()
                            if cond_today_up and cond_past_down: is_treasure = True

                        is_burst = False
                        if not pd.isna(vol_ma5) and vol_ma5 > 0:
                            if vol > (vol_ma5 * 1.5) and price > open_p:
                                is_burst = True

                        streak_days = 0
                        try:
                            for k in range(60):
                                check_idx = -1 - k
                                if abs(check_idx) > len(df_c[ticker]): break
                                
                                if df_c[ticker].iloc[check_idx] > ma200_df[ticker].iloc[check_idx]:
                                    streak_days += 1
                                else:
                                    break 
                        except:
                            streak_days = 0

                        stock_df = pd.DataFrame({'Close': df_c[ticker], 'High': df_h[ticker], 'Low': df_l[ticker]}).dropna()
                        k_val, d_val = 0.0, 0.0
                        if len(stock_df) >= 9:
                            k_val, d_val = calculate_kd_values(stock_df)

                        bias = ((price - ma200) / ma200) * 100
                        stock_info = stock_dict.get(ticker)
                        if not stock_info: continue

                        industry = stock_info.get('group', '其他')

                        raw_data_list.append({
                            '代號': stock_info['code'],
                            '名稱': stock_info['name'],
                            '產業': industry, 
                            '完整代號': ticker,
                            '收盤價': float(price),
                            '生命線': float(ma200),
                            'MA20': float(ma20),
                            'MA60': float(ma60),
                            '生命線趨勢': ma_trend,
                            '乖離率(%)': float(bias),
                            'abs_bias': abs(float(bias)),
                            '成交量': int(vol),
                            '昨日成交量': int(prev_vol),
                            'K值': float(k_val),
                            'D值': float(d_val),
                            '位置': "🟢生命線上" if price >= ma200 else "🔴生命線下",
                            '浴火重生': is_treasure,
                            '爆量起漲': is_burst,
                            '站上天數': int(streak_days)
                        })
                    except: continue
        except Exception: 
            time.sleep(0.2) 
            pass
        
        time.sleep(0.3)
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"努力挖掘中 (Batch=50)...({int(current_progress*100)}%)")
    
    df_result = pd.DataFrame(raw_data_list)
    if not df_result.empty:
        df_result = df_result.drop_duplicates(subset=['完整代號']) 
    
    return df_result

def plot_stock_chart(ticker, name):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df[df['Volume'] > 0].dropna()
        if df.empty:
            st.error("無法取得有效數據")
            return

        df['200MA'] = df['Close'].rolling(window=200).mean()
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['60MA'] = df['Close'].rolling(window=60).mean()
        
        plot_df = df.tail(120).copy()
        plot_df['DateStr'] = plot_df.index.strftime('%Y-%m-%d')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['Close'], mode='lines', name='收盤價', line=dict(color='#00CC96', width=2.5)))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['20MA'], mode='lines', name='20MA(月線)', line=dict(color='#AB63FA', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['60MA'], mode='lines', name='60MA(季線)', line=dict(color='#19D3F3', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['200MA'], mode='lines', name='200MA(生命線)', line=dict(color='#FFA15A', width=3)))

        fig.update_layout(
            title=f"📊 {name} ({ticker}) 股價 vs 均線排列", 
            yaxis_title='價格', 
            height=500, 
            hovermode="x unified",
            xaxis=dict(type='category', tickangle=-45, nticks=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"繪圖失敗: {e}")

# --- 3. 介面顯示區 ---
st.title(f"🍍 {VER} 旺來-台股生命線")
st.markdown("---")

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = None
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None
if 'backtest_result' not in st.session_state:
    st.session_state['backtest_result'] = None
if 'weekly_report' not in st.session_state:
    st.session_state['weekly_report'] = None

with st.sidebar:
    st.header("資料庫管理")
    CACHE_FILE = "stock_data_cache.csv"

    if st.button("🚨 強制重置系統"):
        st.cache_data.clear()
        st.session_state.clear()
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE) 
        st.success("系統已重置！請重新點擊更新股價。")
        st.rerun()

    if st.session_state['master_df'] is None and os.path.exists(CACHE_FILE):
        try:
            df_cache = pd.read_csv(CACHE_FILE)
            if '爆量起漲' not in df_cache.columns: df_cache['爆量起漲'] = False
            if '站上天數' not in df_cache.columns: df_cache['站上天數'] = 0 
            if '產業' not in df_cache.columns: df_cache['產業'] = "未知(請更新)"
                
            st.session_state['master_df'] = df_cache
            mod_time = os.path.getmtime(CACHE_FILE)
            st.session_state['last_update'] = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"⚡ 已快速載入上次資料 ({st.session_state['last_update']})")
        except Exception as e:
            st.error(f"讀取快取失敗: {e}")

    if st.button("🔄 下載最新股價 (開市用)", type="primary"):
        stock_dict = get_stock_list()
        if not stock_dict:
            st.error("無法取得股票清單，請稍後再試或按上方重置按鈕。")
        else:
            placeholder_emoji = st.empty() 
            with placeholder_emoji:
                st.markdown("""<div style="text-align: center; font-size: 40px; animation: blink 1s infinite;">🎁💰✨</div>
                    <style>@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }</style>
                    <div style="text-align: center;">連線下載中 (Batch=50)...</div>""", unsafe_allow_html=True)
            
            st.caption("ℹ️ 已加速下載流程 (Batch=50)，請耐心等候...")
            status_text = st.empty()
            progress_bar = st.progress(0, text="準備下載...")
            df = fetch_all_data(stock_dict, progress_bar, status_text)
            
            if not df.empty:
                df.to_csv(CACHE_FILE, index=False)
                st.session_state['master_df'] = df
                st.session_state['last_update'] = get_taiwan_time_str()
                st.success(f"更新完成！共 {len(df)} 檔資料")
            else:
                st.error("⛔ 資料下載不完整，請稍後再試。")
                with st.expander("🆘 嘗試解決方案：Reboot App (點我展開)"):
                    st.info("""
                    **請嘗試「重啟應用程式」來更換連線環境：**
                    1. 點擊網頁右上角的 **「⋮」** (三個點按鈕)。
                    2. 選擇 **「Reboot App」** (或 Clear Cache and Rerun)。
                    3. 等待網頁重新載入後，再試一次。
                    """)
            
            placeholder_emoji.empty()
            progress_bar.empty()
        
    if st.session_state['last_update']:
        st.caption(f"最後更新：{st.session_state['last_update']}")
    
    st.divider()

    st.header("2. 即時篩選器")
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 2.5, step=0.1)
    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    
    st.subheader("策略選擇")
    strategy_mode = st.radio("選擇篩選策略：", ("🛡️ 守護生命線 (反彈/支撐)", "🔥 浴火重生 (假跌破)"))

    st.caption("基礎條件：")
    col1, col2 = st.columns(2)
    with col1: filter_trend_up = st.checkbox("生命線向上", value=False)
    with col2: filter_trend_down = st.checkbox("生命線向下", value=False)
    filter_kd = st.checkbox("KD 黃金交叉 (今日 K > D)", value=False)
    filter_vol_double = st.checkbox("出量 (今日 > 昨日x1.5)", value=False)
    
    st.markdown("---")
    st.caption("🧪 實驗室 (測試中 - 模擬法人起漲)：")
    filter_burst_vol = st.checkbox("🔥 爆量起漲 (量>5日均量1.5倍 + 紅K)", value=False, help="模擬主力或法人進場訊號")

    if strategy_mode == "🔥 浴火重生 (假跌破)":
        st.info("ℹ️ 尋找：過去7日內曾跌破，但今日站回生命線的個股。")

    st.divider()
    
    st.subheader("3. 策略驗證與戰報")
    st.caption("⚠️ 將使用上方設定的「策略模式」與「乖離率」進行驗證。")
    
    col_bt1, col_bt2 = st.columns(2)
    
    with col_bt1:
        if st.button("🧪 策略回測 (含週報觀點)", use_container_width=True):
            if st.session_state['master_df'] is None:
                st.error("⛔ 請先點擊上方「🔄 下載最新股價」按鈕！")
            else:
                st.info("正在調閱過去半年的歷史檔案，進行深度驗證... ⏳")
                stock_dict = get_stock_list()
                bt_progress = st.progress(0, text="初始化回測...")
                
                use_treasure_param = True if strategy_mode == "🔥 浴火重生 (假跌破)" else False
                
                bt_df = run_strategy_backtest(
                    stock_dict, 
                    bt_progress, 
                    use_trend_up=filter_trend_up, 
                    use_treasure=use_treasure_param, 
                    use_vol=filter_vol_double,
                    min_vol_threshold=min_vol_input,
                    use_burst_vol=filter_burst_vol
                )
                
                st.session_state['backtest_result'] = bt_df
                st.session_state['weekly_report'] = None 
                bt_progress.empty()
                st.rerun()

    with col_bt2:
        if st.button("📊 生成本週戰報 (測試中)", use_container_width=True):
            if st.session_state['master_df'] is None:
                st.error("⛔ 請先點擊上方「🔄 下載最新股價」按鈕！")
            else:
                stock_dict = get_stock_list()
                scan_progress = st.progress(0, text="戰情室連線中...")
                
                # 關鍵修正：將側邊欄所有條件傳入
                df_scan = scan_period_signals(
                    stock_dict, 
                    5, 
                    scan_progress, 
                    min_vol_input,
                    bias_threshold,
                    strategy_mode,
                    filter_trend_up, # New
                    filter_trend_down, # New
                    filter_kd, # New
                    filter_vol_double, # New
                    filter_burst_vol # New
                )
                st.session_state['weekly_report'] = df_scan
                st.session_state['backtest_result'] = None
                scan_progress.empty()
                st.rerun()

    with st.expander("📅 系統開發日誌"):
        st.write(f"**🕒 系統最後重啟時間:** {get_taiwan_time_str()}")
        st.markdown("---")
        st.markdown("""
        ### Ver 5.1 (Logic Synced)
        * **Fix**: **週報邏輯同步** - 修正週報只看乖離率的問題。現在週報會同步套用「生命線趨勢」、「KD」、「成交量」等所有側邊欄條件，結果將與今日篩選一致。
        * **UI**: **戰情堆疊表** - 週報改為列表顯示，依照日期排序，清楚展示本週每日的符合個股。
        * **Fix**: **回測數據校正** - 修正回測指標數字固定的問題，現在會依據表格內容即時計算。
        """)
    
    st.divider()
    with st.expander("🔐 管理員後台"):
        admin_pwd = st.text_input("請輸入管理密碼", type="password")
        if admin_pwd == "admin888": 
            if os.path.exists(LOG_FILE):
                st.markdown("### 🚦 流量統計 (最近紀錄)")
                log_df = pd.read_csv(LOG_FILE)
                total_visits = len(log_df)
                unique_users = log_df['Session_ID'].nunique()
                st.metric("總點擊次數", total_visits)
                st.metric("獨立訪客數 (Session)", unique_users)
                st.dataframe(log_df.sort_values(by="時間", ascending=False), use_container_width=True)
                with open(LOG_FILE, "rb") as f:
                    st.download_button("📥 下載完整 Log (CSV)", f, file_name="traffic_log.csv", mime="text/csv")
            else:
                st.info("尚無流量紀錄。")
        elif admin_pwd:
            st.error("密碼錯誤")

# ==========================================
# 4. 顯示邏輯：區塊堆疊 (不再互斥)
# ==========================================

# (A) 預設歡迎畫面 (若完全沒資料)
if st.session_state['master_df'] is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; font-size: 1.2em; line-height: 2.0; color: #555; margin-bottom: 15px;">
            這是數年來的經驗收納<br>
            此工具僅供參考，不代表投資建議<br>
            <span style="font-size: 1.3em; color: #6a0dad; font-weight: bold;">預祝心想事成，從從容容，紫氣東來! 🟣✨</span><br>
            <span style="font-size: 0.9em; color: #888;">程式已達千行!越來越強大啦! 🚀</span>
        </div>
        """, unsafe_allow_html=True)

        if os.path.exists("welcome.jpg"):
            st.image("welcome.jpg", width=180)
            
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 下載最新股價」** 按鈕開始挖寶！")

# (B) 今日篩選結果 (永遠顯示在最上方)
if st.session_state['master_df'] is not None:
    df = st.session_state['master_df'].copy()
    
    # 篩選邏輯
    if '生命線' not in df.columns:
        st.error("⚠️ 資料結構已更新！請重置系統。")
        st.stop()

    df = df[df['成交量'] >= (min_vol_input * 1000)]
    
    if strategy_mode == "🔥 浴火重生 (假跌破)":
        df = df[df['浴火重生'] == True]
    else:
        df = df[df['abs_bias'] <= bias_threshold]
        if filter_trend_up: df = df[df['生命線趨勢'] == "⬆️向上"]
        elif filter_trend_down: df = df[df['生命線趨勢'] == "⬇️向下"]
        if filter_kd: df = df[df['K值'] > df['D值']]
    
    if filter_vol_double: 
        df = df[df['成交量'] > (df['昨日成交量'] * 1.5)]
    
    if filter_burst_vol:
        if '爆量起漲' in df.columns:
            df = df[df['爆量起漲'] == True]
        
    if len(df) == 0:
        st.warning(f"⚠️ 找不到符合條件的股票！")
    else:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #ff4b4b;">
            <h2 style="color: #333; margin:0;">🔍 根據【{strategy_mode}】，共篩選出 <span style="color: #ff4b4b; font-size: 1.5em;">{len(df)}</span> 檔股票</h2>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        df['成交量(張)'] = (df['成交量'] / 1000).astype(int)
        df['KD值'] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
        df['選股標籤'] = df['代號'].astype(str) + " " + df['名稱'].astype(str)
        df['法人買賣?'] = df['代號'].apply(lambda x: f"https://tw.stock.yahoo.com/quote/{x}/institutional-trading")

        display_cols = ['代號', '名稱', '產業', '收盤價', '生命線', '站上天數', '乖離率(%)', 'KD值', '成交量(張)', '法人買賣?']
        df = df.sort_values(by='成交量', ascending=False)
        
        tab1, tab2 = st.tabs(["📋 今日篩選結果", "📊 個股趨勢圖"])
        
        with tab1:
            def highlight_row(row):
                return ['background-color: #e6fffa; color: black'] * len(row) if row['收盤價'] > row['生命線'] else ['background-color: #fff0f0; color: black'] * len(row)

            st.dataframe(
                df[display_cols].style.apply(highlight_row, axis=1),
                use_container_width=True, hide_index=True,
                column_config={
                    "法人買賣?": st.column_config.LinkColumn("🔍 查法人", display_text="前往查看")
                }
            )
            st.info("💡 想要知道這些股票過去的勝率如何？請點擊下方的「🧪 策略回測」或「📊 生成本週戰報」。")

        with tab2:
            st.markdown("### 🔍 個股趨勢圖")
            if len(df) > 0:
                selected_stock_label = st.selectbox("請選擇一檔股票：", df['選股標籤'].tolist())
                selected_row = df[df['選股標籤'] == selected_stock_label].iloc[0]
                plot_stock_chart(selected_row['完整代號'], selected_row['名稱'])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("收盤價", f"{selected_row['收盤價']:.2f}")
                c2.metric("成交量", f"{selected_row['成交量(張)']} 張")
                c3.metric("KD", selected_row['KD值'])

# (C) 週報戰情室 (堆疊表格顯示)
if st.session_state['weekly_report'] is not None:
    df_scan = st.session_state['weekly_report']
    st.markdown("---")
    st.subheader(f"📊 本週戰報：{strategy_mode}")
    
    if not df_scan.empty:
        df_scan = df_scan.sort_values(by=['訊號日期', '至今漲跌(%)'], ascending=[False, False])
        st.markdown("#### 📝 戰情堆疊表 (依日期排序)")
        
        # 顯示完整堆疊表格
        st.dataframe(
            df_scan[['訊號日期', '距今', '代號', '名稱', '產業', '訊號價', '現價', '至今漲跌(%)', '站穩', '狀態']],
            use_container_width=True,
            column_config={
                "至今漲跌(%)": st.column_config.ProgressColumn(
                    "損益表現", format="%.2f%%", min_value=-10, max_value=10
                ),
                "站穩": st.column_config.NumberColumn("站穩(天)")
            },
            hide_index=True
        )
    else:
        st.warning("🧐 過去 5 天內沒有發現符合目前篩選條件的股票。")

# (D) 策略回測報告 (動態指標修正)
if st.session_state['backtest_result'] is not None:
    bt_df = st.session_state['backtest_result']
    st.markdown("---")
    
    s_name = "🛡️ 守護生命線"
    if strategy_mode == "🔥 浴火重生 (假跌破)": s_name = "🔥 浴火重生"
    
    st.subheader(f"🧪 策略回測報告：{s_name}")
    bt_df['訊號日期'] = pd.to_datetime(bt_df['訊號日期'])
    
    # 戰略週報圖表
    if not bt_df.empty:
        bt_df['週次'] = bt_df['訊號日期'] - pd.to_timedelta(bt_df['訊號日期'].dt.dayofweek, unit='d')
        weekly_stats = bt_df.groupby('週次').agg(
            總訊號數=('代號', 'count'),
            勝場數=('is_win', 'sum')
        ).reset_index()
        weekly_stats['勝率'] = (weekly_stats['勝場數'] / weekly_stats['總訊號數'] * 100).round(1)
        weekly_stats['週次字串'] = weekly_stats['週次'].dt.strftime('%m/%d')
        weekly_stats = weekly_stats.sort_values('週次')

        st.markdown("#### 🏆 戰略週報：勝率趨勢圖")
        try:
            fig_week = go.Figure()
            fig_week.add_trace(go.Bar(
                x=weekly_stats['週次字串'], y=weekly_stats['總訊號數'], name='訊號數量', marker_color='rgba(50, 171, 96, 0.6)', yaxis='y2'
            ))
            fig_week.add_trace(go.Scatter(
                x=weekly_stats['週次字串'], y=weekly_stats['勝率'], name='勝率(%)', mode='lines+markers', line=dict(color='#FF5733', width=3)
            ))
            fig_week.update_layout(
                title='每週 訊號數量 vs 勝率', template='plotly_white', xaxis=dict(title='週次'),
                yaxis=dict(title='勝率 (%)', title_font=dict(color='#FF5733'), range=[0, 105]),
                yaxis2=dict(title='訊號數量', title_font=dict(color='rgba(50, 171, 96, 0.6)'), overlaying='y', side='right', showgrid=False),
                hovermode="x unified", legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_week, use_container_width=True)
        except Exception: pass
        
    st.markdown("---")

    # 歷史列表
    bt_df['訊號日期_str'] = bt_df['訊號日期'].dt.strftime('%Y-%m-%d')
    df_history = bt_df[bt_df['結果'] != "觀察中"].copy()
    df_watching = bt_df[bt_df['結果'] == "觀察中"].copy()

    # 1. 顯示未結算 (觀察中) 股票
    if not df_watching.empty:
        st.markdown(f"""
        <div style="background-color: #fff8dc; padding: 15px; border-radius: 10px; border: 2px solid #ffa500; margin-bottom: 20px;">
            <h3 style="color: #d2691e; margin:0;">👀 旺來關注中 (進行中訊號)</h3>
            <p style="color: #666; margin:5px 0 0 0;">這些股票訊號發生未滿 10 天，尚未結算。</p>
        </div>
        """, unsafe_allow_html=True)
        
        df_watching = df_watching.sort_values(by='訊號日期', ascending=False)
        st.dataframe(
            df_watching[['代號', '名稱', '產業', '訊號日期_str', '訊號價', '最高漲幅(%)']].style.background_gradient(cmap='Reds', subset=['最高漲幅(%)']),
            use_container_width=True, hide_index=True
        )

    # 2. 顯示已結算歷史數據
    st.markdown("### 📜 歷史驗證數據 (已結算)")
    
    if len(df_history) > 0:
        # 動態計算指標 (Critical Fix)
        total_count = len(df_history)
        win_df = df_history[df_history['結果'].str.contains("Win") | df_history['結果'].str.contains("驗證成功")]
        win_count = len(win_df)
        win_rate = int((win_count / total_count) * 100) if total_count > 0 else 0
        avg_max_ret = round(df_history['最高漲幅(%)'].mean(), 2)
        
        # 顯示指標 (移到 Tab 上方)
        c1, c2, c3 = st.columns(3)
        c1.metric("總已結算", total_count)
        c2.metric("獲利機率", f"{win_rate}%")
        c3.metric("平均損益", f"{avg_max_ret}%")

        months = sorted(df_history['月份'].unique())
        tabs = st.tabs(["📊 總覽"] + months)
        
        with tabs[0]:
            st.dataframe(df_history[['月份', '代號', '名稱', '訊號日期_str', '訊號價', '最高漲幅(%)', '結果']], use_container_width=True)
            
        for i, m in enumerate(months):
            with tabs[i+1]:
                m_df = df_history[df_history['月份'] == m]
                def color_ret(val): return f'color: {"red" if val > 0 else "green"}'
                st.dataframe(m_df[['代號', '名稱', '訊號日期_str', '訊號價', '最高漲幅(%)', '結果']].style.map(color_ret, subset=['最高漲幅(%)']), use_container_width=True)
    else:
        st.warning("在此回測期間內，沒有歷史股票符合條件。")
