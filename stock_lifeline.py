import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime
import plotly.graph_objects as go
import requests
import os

# --- 1. 網頁設定 ---
VER = "ver3.9"
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

# --- 策略回測核心函數 ---
def run_strategy_backtest(stock_dict, progress_bar):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
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
                scan_window = df_c.index[-90:-10] 
                
                for ticker in df_c.columns:
                    try:
                        c_series = df_c[ticker]
                        v_series = df_v[ticker]
                        l_series = df_l[ticker]
                        h_series = df_h[ticker]
                        ma_series = ma200_df[ticker]
                        
                        stock_name = stock_dict.get(ticker, {}).get('name', ticker)
                        
                        for date in scan_window:
                            if pd.isna(ma_series[date]): continue
                            
                            idx = c_series.index.get_loc(date)
                            if idx < 20: continue 

                            close_p = c_series.iloc[idx]
                            low_p = l_series.iloc[idx]
                            vol = v_series.iloc[idx]
                            prev_vol = v_series.iloc[idx-1]
                            ma_val = ma_series.iloc[idx]
                            ma_val_20ago = ma_series.iloc[idx-20]
                            
                            if ma_val == 0 or prev_vol == 0: continue

                            cond_near = (low_p <= ma_val * 1.03) and (low_p >= ma_val * 0.90) 
                            cond_vol = (vol > prev_vol * 1.5)
                            cond_up = (close_p > ma_val)
                            cond_trend = (ma_val > ma_val_20ago)
                            
                            if cond_near and cond_vol and cond_up and cond_trend:
                                future_highs = h_series.iloc[idx+1 : idx+11]
                                max_price = future_highs.max()
                                max_profit_pct = (max_price - close_p) / close_p * 100
                                
                                month_str = date.strftime('%m月')
                                
                                is_win = max_profit_pct >= 3.0
                                
                                results.append({
                                    '月份': month_str,
                                    '代號': ticker.replace(".TW", "").replace(".TWO", ""),
                                    '名稱': stock_name,
                                    '訊號日期': date.strftime('%Y-%m-%d'),
                                    '訊號價': round(close_p, 2),
                                    '最高漲幅(%)': round(max_profit_pct, 2),
                                    '結果': "Win 🏆" if is_win else "Loss 📉"
                                })
                                break 
                    except:
                        continue
        except:
            pass
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中 (計算分月數據)...({int(progress*100)}%)")
        
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
                last_price_series = df_c.iloc[-1]
                last_ma200_series = ma200_df.iloc[-1]
                prev_ma200_series = ma200_df.iloc[-21] 
                
                last_vol_series = df_v.iloc[-1]
                prev_vol_series = df_v.iloc[-2]

                recent_close_df = df_c.iloc[-8:]
                recent_ma200_df = ma200_df.iloc[-8:]

                for ticker in df_c.columns:
                    try:
                        price = last_price_series[ticker]
                        ma200 = last_ma200_series[ticker]
                        prev_ma200 = prev_ma200_series[ticker]
                        
                        vol = last_vol_series[ticker]
                        prev_vol = prev_vol_series[ticker]
                        
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
                            '收盤價': float(price),
                            '生命線': float(ma200),
                            '生命線趨勢': ma_trend,
                            '乖離率(%)': float(bias),
                            'abs_bias': abs(float(bias)),
                            '成交量': int(vol),
                            '昨日成交量': int(prev_vol),
                            'K值': float(k_val),
                            'D值': float(d_val),
                            '位置': "🟢生命線上" if price >= ma200 else "🔴生命線下",
                            '浴火重生': is_treasure
                        })
                    except: continue
        except: pass
        
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"系統正在努力挖掘寶藏中...({int(current_progress*100)}%)")
        time.sleep(0.05)
    
    return pd.DataFrame(raw_data_list)

def plot_stock_chart(ticker, name):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df[df['Volume'] > 0].dropna()
        if df.empty:
            st.error("無法取得有效數據")
            return

        df['200MA'] = df['Close'].rolling(window=200).mean()
        
        plot_df = df.tail(120).copy()
        plot_df['DateStr'] = plot_df.index.strftime('%Y-%m-%d')

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_df['DateStr'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'],
            name='日收盤價', increasing_line_color='red', decreasing_line_color='green'
        ))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['200MA'], line=dict(color='orange', width=2), name='生命線'))

        fig.update_layout(
            title=f"📊 {name} ({ticker}) 近半年日K線圖", yaxis_title='股價', height=600, hovermode="x unified",
            xaxis=dict(type='category', tickangle=-45, nticks=20), xaxis_rangeslider_visible=False
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

with st.sidebar:
    st.header("資料庫管理")
    
    if st.button("🚨 強制重置系統"):
        st.cache_data.clear()
        st.session_state.clear()
        st.success("系統已重置！請重新點擊更新股價。")
        st.rerun()

    if st.button("🔄 更新股價資料 (開市請按我)", type="primary"):
        stock_dict = get_stock_list()
        
        if not stock_dict:
            st.error("無法取得股票清單，請稍後再試或按上方重置按鈕。")
        else:
            placeholder_emoji = st.empty() 
            with placeholder_emoji:
                st.markdown("""
                    <div style="text-align: center; font-size: 40px; animation: blink 1s infinite;">
                        🎁💰✨
                    </div>
                    <style>
                    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
                    </style>
                    <div style="text-align: center;">正在開鎖寶箱...</div>
                """, unsafe_allow_html=True)
            
            status_text = st.empty()
            progress_bar = st.progress(0, text="準備下載...")
            
            df = fetch_all_data(stock_dict, progress_bar, status_text)
            
            placeholder_emoji.empty()
            
            st.session_state['master_df'] = df
            st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            progress_bar.empty()
            st.success(f"更新完成！共 {len(df)} 檔資料")
        
    if st.session_state['last_update']:
        st.caption(f"最後更新：{st.session_state['last_update']}")
    
    st.divider()
    st.header("2. 即時篩選器")
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 5.0, 2.5, step=0.1)
    st.caption("設定股價距離「生命線」多近視為符合條件。")
    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    
    st.subheader("進階條件")
    
    filter_trend_up = st.checkbox("📈 生命線向上 (多方助漲)", value=False)
    filter_trend_down = st.checkbox("📉 生命線向下
