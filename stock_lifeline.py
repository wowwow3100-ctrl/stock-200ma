import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import os

# --- 1. 網頁設定 ---
VER = "ver4.9_SmartMoney"
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
            if info.type == '股票':
                if info.group not in exclude_industries:
                    stock_dict[f"{code}.TW"] = {'name': info.name, 'code': code, 'group': info.group}
                    
        for code, info in otc.items():
            if info.type == '股票':
                if info.group not in exclude_industries:
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
    """計算 OBV 能量潮 (大戶籌碼代理指標)"""
    try:
        # OBV = 累積 (如果收盤漲 sign=1 * Vol, 跌 sign=-1 * Vol)
        # 填補 0 避免計算錯誤
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv
    except:
        return pd.Series(0, index=df.index)

# --- 策略最佳化擂台函數 ---
def run_optimization_tournament(stock_dict, progress_bar):
    raw_signals = [] 
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            
            # 處理 MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                pass 
            
            if not data.empty:
                try:
                    df_c = data['Close']
                    df_v = data['Volume']
                    df_l = data['Low']
                    df_h = data['High']
                except KeyError:
                    continue
                
                if isinstance(df_c, pd.Series):
                    ticker = batch[0]
                    df_c = df_c.to_frame(name=ticker)
                    df_v = df_v.to_frame(name=ticker)
                    df_l = df_l.to_frame(name=ticker)
                    df_h = df_h.to_frame(name=ticker)

                ma200_df = df_c.rolling(window=200).mean()
                
                # 計算全體 OBV (稍微複雜因為是 DataFrame)
                # 這裡針對每一欄位跑迴圈計算比較穩當
                obv_df = pd.DataFrame(index=df_c.index, columns=df_c.columns)
                for col in df_c.columns:
                    obv_df[col] = calculate_obv(pd.DataFrame({'Close': df_c[col], 'Volume': df_v[col]}))

                scan_window_idx = df_c.index[-250:-20] 
                
                for ticker in df_c.columns:
                    try:
                        c_series = df_c[ticker]
                        v_series = df_v[ticker]
                        l_series = df_l[ticker]
                        ma_series = ma200_df[ticker]
                        obv_series = obv_df[ticker]
                        
                        if c_series.isna().sum() > 100 or ma_series.isna().all(): continue

                        for date in scan_window_idx:
                            if pd.isna(ma_series[date]): continue
                            
                            idx = c_series.index.get_loc(date)
                            if idx < 20: continue 

                            close_p = float(c_series.iloc[idx])
                            low_p = float(l_series.iloc[idx])
                            vol = float(v_series.iloc[idx])
                            prev_vol = float(v_series.iloc[idx-1])
                            ma_val = float(ma_series.iloc[idx])
                            ma_val_20ago = float(ma_series.iloc[idx-20])
                            
                            if ma_val == 0 or prev_vol == 0: continue

                            # --- 1. 基礎訊號 ---
                            cond_near = (low_p <= ma_val * 1.03) and (low_p >= ma_val * 0.90)
                            cond_up = (close_p > ma_val)
                            is_basic_signal = cond_near and cond_up
                            
                            # --- 2. 特徵標記 ---
                            tag_trend_up = (ma_val > ma_val_20ago)
                            tag_vol_double = (vol > prev_vol * 1.5)
                            
                            # 籌碼標記 (Smart Money)：前一週 (5天前) OBV 是否低於現在 (代表這一週在吸籌)
                            obv_now = obv_series.iloc[idx]
                            obv_week_ago = obv_series.iloc[idx-5]
                            tag_obv_in = obv_now > obv_week_ago

                            # 浴火重生
                            tag_treasure = False
                            start_idx = idx - 7
                            if start_idx >= 0:
                                recent_c = c_series.iloc[start_idx : idx+1]
                                recent_ma = ma_series.iloc[start_idx : idx+1]
                                cond_today_up = recent_c.iloc[-1] > recent_ma.iloc[-1]
                                cond_past_down = (recent_c.iloc[:-1] < recent_ma.iloc[:-1]).any()
                                if cond_today_up and cond_past_down:
                                    tag_treasure = True

                            if not is_basic_signal and not tag_treasure:
                                continue
                                
                            # --- 3. 結果驗證 ---
                            if idx + 20 < len(c_series):
                                future_close = float(c_series.iloc[idx + 20])
                                profit_pct = (future_close - close_p) / close_p * 100
                                is_win = profit_pct > 0 
                            else:
                                continue 

                            raw_signals.append({
                                'Ticker': ticker,
                                'Date': date,
                                'Profit_Pct': profit_pct,
                                'Is_Win': is_win,
                                'Tag_Trend_Up': tag_trend_up,
                                'Tag_Vol_Double': tag_vol_double,
                                'Tag_Treasure': tag_treasure,
                                'Tag_OBV_In': tag_obv_in, # 新增
                                'Is_Basic_Near': is_basic_signal
                            })

                    except Exception:
                        continue
        except:
            pass
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"全策略掃描中...({int(progress*100)}%)")
        
    return pd.DataFrame(raw_signals)

# --- 單一回測函數 (維持原樣，僅加入參數接收) ---
def run_strategy_backtest(stock_dict, progress_bar, use_trend_up, use_treasure, use_vol, use_obv):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                pass

            if not data.empty:
                try:
                    df_c = data['Close']
                    df_v = data['Volume']
                    df_l = data['Low']
                    df_h = data['High']
                except KeyError:
                    continue
                
                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])

                ma200_df = df_c.rolling(window=200).mean()
                
                # 計算 OBV
                obv_df = pd.DataFrame(index=df_c.index, columns=df_c.columns)
                for col in df_c.columns:
                    obv_df[col] = calculate_obv(pd.DataFrame({'Close': df_c[col], 'Volume': df_v[col]}))

                scan_window = df_c.index[-250:-20] 
                
                for ticker in df_c.columns:
                    try:
                        c_series = df_c[ticker]
                        v_series = df_v[ticker]
                        l_series = df_l[ticker]
                        ma_series = ma200_df[ticker]
                        obv_series = obv_df[ticker]
                        
                        stock_name = stock_dict.get(ticker, {}).get('name', ticker)
                        stock_code = stock_dict.get(ticker, {}).get('code', ticker.split('.')[0])
                        
                        for date in scan_window:
                            if pd.isna(ma_series[date]): continue
                            
                            idx = c_series.index.get_loc(date)
                            if idx < 20: continue 

                            close_p = float(c_series.iloc[idx])
                            low_p = float(l_series.iloc[idx])
                            vol = float(v_series.iloc[idx])
                            prev_vol = float(v_series.iloc[idx-1])
                            ma_val = float(ma_series.iloc[idx])
                            ma_val_20ago = float(ma_series.iloc[idx-20])
                            
                            # OBV Check
                            obv_now = obv_series.iloc[idx]
                            obv_week_ago = obv_series.iloc[idx-5]
                            is_obv_up = obv_now > obv_week_ago

                            if ma_val == 0 or prev_vol == 0: continue

                            is_match = False
                            
                            # 過濾條件
                            if use_trend_up and (ma_val <= ma_val_20ago): continue
                            if use_vol and (vol <= prev_vol * 1.5): continue
                            if use_obv and not is_obv_up: continue # OBV 濾網

                            if use_treasure:
                                start_idx = idx - 7
                                if start_idx < 0: continue
                                recent_c = c_series.iloc[start_idx : idx+1]
                                recent_ma = ma_series.iloc[start_idx : idx+1]
                                cond_today_up = recent_c.iloc[-1] > recent_ma.iloc[-1]
                                past_c = recent_c.iloc[:-1]
                                past_ma = recent_ma.iloc[:-1]
                                cond_past_down = (past_c < past_ma).any()
                                if cond_today_up and cond_past_down: is_match = True
                            else:
                                cond_near = (low_p <= ma_val * 1.03) and (low_p >= ma_val * 0.90) 
                                cond_up = (close_p > ma_val)
                                if cond_near and cond_up: is_match = True
                            
                            if is_match:
                                if idx + 20 < len(c_series):
                                    future_close = float(c_series.iloc[idx + 20])
                                    profit_pct = (future_close - close_p) / close_p * 100
                                    
                                    if profit_pct > 0:
                                        result_status = "Win (上漲)"
                                    else:
                                        result_status = "Loss (下跌)"
                                else:
                                    profit_pct = np.nan
                                    result_status = "統計中"

                                month_str = date.strftime('%Y-%m')
                                
                                results.append({
                                    '月份': month_str,
                                    'StockID': stock_code,
                                    '名稱': stock_name,
                                    'Date': date,
                                    '訊號日期': date.strftime('%Y-%m-%d'),
                                    '訊號價': round(float(close_p), 2),
                                    '未來20日收盤': round(float(future_close), 2) if not np.isnan(profit_pct) else np.nan,
                                    '一個月內漲幅(%)': round(float(profit_pct), 2) if not np.isnan(profit_pct) else np.nan,
                                    '結果': result_status
                                })
                    except Exception as e:
                        continue
        except:
            pass
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中...({int(progress*100)}%)")
        
    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    def generate_log(row):
        rise_pct = f"{row['一個月內漲幅(%)']}%" if not pd.isna(row['一個月內漲幅(%)']) else "統計中"
        return (f"日期: {row['訊號日期']} | "
                f"股票: {row['StockID']} | "
                f"觸發價: {row['訊號價']} | "
                f"後續漲幅: {rise_pct}")

    df_results['紀錄日誌'] = df_results.apply(generate_log, axis=1)
    df_results = df_results.sort_values(by=['Date', 'StockID'], ascending=[False, True])
    
    return df_results

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
            if isinstance(data.columns, pd.MultiIndex):
                pass
            
            if not data.empty:
                try:
                    df_c = data['Close']
                    df_h = data['High']
                    df_l = data['Low']
                    df_v = data['Volume']
                except KeyError:
                    continue

                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])

                ma200_df = df_c.rolling(window=200).mean()
                
                # 計算 OBV (Real-time)
                obv_df = pd.DataFrame(index=df_c.index, columns=df_c.columns)
                for col in df_c.columns:
                    obv_df[col] = calculate_obv(pd.DataFrame({'Close': df_c[col], 'Volume': df_v[col]}))
                
                last_price_series = df_c.iloc[-1]
                last_ma200_series = ma200_df.iloc[-1]
                prev_ma200_series = ma200_df.iloc[-21] 
                
                last_vol_series = df_v.iloc[-1]
                prev_vol_series = df_v.iloc[-2]

                recent_close_df = df_c.iloc[-8:]
                recent_ma200_df = ma200_df.iloc[-8:]

                for ticker in df_c.columns:
                    try:
                        price = float(last_price_series[ticker])
                        ma200 = float(last_ma200_series[ticker])
                        prev_ma200 = float(prev_ma200_series[ticker])
                        vol = float(last_vol_series[ticker])
                        prev_vol = float(prev_vol_series[ticker])
                        
                        # OBV Check
                        obv_series = obv_df[ticker]
                        obv_now = obv_series.iloc[-1]
                        obv_week_ago = obv_series.iloc[-6] # 比較 5 天前
                        is_obv_in = obv_now > obv_week_ago
                        
                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0: continue

                        ma_trend = "⬆️向上" if ma200 >= prev_ma200 else "⬇️向下"

                        is_treasure = False
                        my_recent_c = recent_close_df[ticker]
                        my_recent_ma = recent_ma200_df[ticker]
                        
                        if len(my_recent_c) >= 8:
                            cond_today_up = float(my_recent_c.iloc[-1]) > float(my_recent_ma.iloc[-1])
                            past_c = my_recent_c.iloc[:-1]
                            past_ma = my_recent_ma.iloc[:-1]
                            cond_past_down = (past_c < past_ma).any()
                            
                            if cond_today_up and cond_past_down:
                                is_treasure = True

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
                            'OBV趨勢': "🔥吸籌" if is_obv_in else "☁️一般"
                        })
                    except Exception as e: 
                        continue
        except: pass
        
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"系統正在努力挖掘寶藏中...({int(current_progress*100)}%)")
        time.sleep(0.02)
    
    return pd.DataFrame(raw_data_list)

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
        # 繪圖時也順便算一下 OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        plot_df = df.tail(120).copy()
        plot_df['DateStr'] = plot_df.index.strftime('%Y-%m-%d')

        fig = go.Figure()
        
        # 股價與MA (主圖)
        fig.add_trace(go.Scatter(
            x=plot_df['DateStr'], y=plot_df['Close'], 
            mode='lines', name='收盤價',
            line=dict(color='#00CC96', width=2.5) 
        ))
        fig.add_trace(go.Scatter(
            x=plot_df['DateStr'], y=plot_df['200MA'], 
            mode='lines', name='生命線',
            line=dict(color='#FFA15A', width=3) 
        ))
        
        # 為了不讓圖表太亂，這裡只顯示股價
        # OBV 通常需要副圖，Streamlit 簡單版先不畫副圖以免手機版跑版
        
        fig.update_layout(
            title=f"📊 {name} ({ticker}) 股價 vs 生命線趨勢", 
            yaxis_title='價格', 
            height=500, 
            hovermode="x unified",
            xaxis=dict(type='category', tickangle=-45, nticks=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 額外顯示 OBV 狀態
        obv_trend = "📈 增加中 (有人在買)" if plot_df['OBV'].iloc[-1] > plot_df['OBV'].iloc[-6] else "📉 持平或減少"
        st.info(f"💡 籌碼雷達 (OBV)：近一週 {obv_trend}")
        
    except Exception as e: st.error(f"繪圖失敗: {e}")

# --- 3. 介面顯示區 ---
st.title(f"🍍 {VER} 旺來-台股生命線")
st.markdown("---")

if 'master_df' not in st.session
