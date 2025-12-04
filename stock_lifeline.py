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
                
                # 計算全體 OBV
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
                            
                            # 籌碼標記 (Smart Money)：前一週 (5天前) OBV 是否低於現在
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
                                'Tag_OBV_In': tag_obv_in,
                                'Is_Basic_Near': is_basic_signal
                            })

                    except Exception:
                        continue
        except:
            pass
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"全策略掃描中...({int(progress*100)}%)")
        
    return pd.DataFrame(raw_signals)

# --- 單一回測函數 ---
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

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = None
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None
if 'backtest_result' not in st.session_state:
    st.session_state['backtest_result'] = None
if 'optimizer_result' not in st.session_state:
    st.session_state['optimizer_result'] = None

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
    st.header("功能選擇")
    
    bias_threshold = st.slider("乖離率範圍 (±%)", 0.5, 20.0, 5.0, step=0.1)
    if bias_threshold <= 5.0:
        st.caption("🛡️ 防守型")
    else:
        st.caption("⚔️ 攻擊型")

    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    
    st.subheader("篩選濾網")
    
    filter_trend_up = st.checkbox("📈 生命線向上 (多方)", value=False)
    filter_trend_down = st.checkbox("📉 生命線向下 (空方)", value=False)
    filter_treasure = st.checkbox("🔥 浴火重生 (假跌破)", value=False)
    filter_obv = st.checkbox("🕵️ 潛伏雷達 (OBV吃貨)", value=False) # New
    filter_kd = st.checkbox("KD 黃金交叉", value=False)
    filter_vol_double = st.checkbox("出量 ( > 昨日x1.5)", value=False)
    
    st.divider()
    
    st.subheader("策略實驗室")
    if st.button("🏆 執行策略擂台 (尋找最強組合)"):
        st.info("正在比較 6 種策略 (含籌碼分析) 的勝率... (約需 2-3 分鐘)")
        stock_dict = get_stock_list()
        opt_progress = st.progress(0, text="初始化擂台...")
        
        opt_df = run_optimization_tournament(stock_dict, opt_progress)
        st.session_state['optimizer_result'] = opt_df
        opt_progress.empty()
        st.success("擂台賽結束！請看右側報告。")

    if st.button("🧪 單一策略回測 (含20日後驗證)"):
        st.info("執行指定條件回測... ⏳")
        stock_dict = get_stock_list()
        bt_progress = st.progress(0, text="初始化回測...")
        
        bt_df = run_strategy_backtest(
            stock_dict, 
            bt_progress, 
            use_trend_up=filter_trend_up, 
            use_treasure=filter_treasure, 
            use_vol=filter_vol_double,
            use_obv=filter_obv # New
        )
        
        st.session_state['backtest_result'] = bt_df
        bt_progress.empty()
        st.success("回測完成！")

    with st.expander("📅 系統開發日誌"):
        st.markdown("""
        ### Ver 4.9 (Smart Money)
        * **New Feature**: 新增 **「OBV 潛伏雷達」**。
        * **Logic**: 由於免費工具無法取得集保戶股權分散表(大戶持股)，改用 OBV (能量潮) 作為大戶進出貨的代理指標。
        * **Criteria**: 偵測訊號觸發前一週 (5個交易日)，OBV 是否呈現淨流入 (吸籌)。
        """)

# --- 主畫面顯示 ---

# 1. 策略擂台結果
if st.session_state['optimizer_result'] is not None:
    df_opt = st.session_state['optimizer_result']
    st.subheader("🏆 策略擂台賽：哪種條件最會漲？")
    st.caption("統計過去 250 個交易日，持有 20 天後的表現。")
    
    if not df_opt.empty:
        strategies = {
            "1. 裸測 (接近生命線)": df_opt[df_opt['Is_Basic_Near'] == True],
            "2. 順勢 (生命線向上)": df_opt[(df_opt['Is_Basic_Near'] == True) & (df_opt['Tag_Trend_Up'] == True)],
            "3. 爆量 (出量攻擊)": df_opt[(df_opt['Is_Basic_Near'] == True) & (df_opt['Tag_Vol_Double'] == True)],
            "4. 浴火重生 (假跌破)": df_opt[df_opt['Tag_Treasure'] == True],
            "5. 黃金組合 (順勢+爆量)": df_opt[(df_opt['Is_Basic_Near'] == True) & (df_opt['Tag_Trend_Up'] == True) & (df_opt['Tag_Vol_Double'] == True)],
            "6. 潛伏雷達 (OBV吃貨)": df_opt[(df_opt['Is_Basic_Near'] == True) & (df_opt['Tag_OBV_In'] == True)], # New
        }
        
        summary_list = []
        for name, sub_df in strategies.items():
            total = len(sub_df)
            if total > 0:
                wins = len(sub_df[sub_df['Is_Win'] == True])
                win_rate = (wins / total) * 100
                avg_profit = sub_df['Profit_Pct'].mean()
                summary_list.append({
                    "策略名稱": name,
                    "交易次數": total,
                    "勝率 (%)": win_rate,
                    "平均報酬 (%)": avg_profit
                })
            else:
                summary_list.append({
                    "策略名稱": name,
                    "交易次數": 0,
                    "勝率 (%)": 0,
                    "平均報酬 (%)": 0
                })
        
        sum_df = pd.DataFrame(summary_list)
        sum_df = sum_df.sort_values(by="勝率 (%)", ascending=False)
        
        st.dataframe(
            sum_df.style.background_gradient(subset=['勝率 (%)', '平均報酬 (%)'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        best_strat = sum_df.iloc[0]
        st.success(f"🎉 目前冠軍策略是：**{best_strat['策略名稱']}** (勝率 {best_strat['勝率 (%)']:.1f}%)")
        st.markdown("---")

# 2. 單一回測報告
if st.session_state['backtest_result'] is not None:
    bt_df = st.session_state['backtest_result']
    
    strategy_name = "自訂策略"
    if filter_obv: strategy_name += " + 籌碼潛伏"
    
    st.subheader(f"🧪 單一策略詳情：{strategy_name}")
    
    if len(bt_df) > 0:
        win_count = len(bt_df[bt_df['結果'].str.contains("Win")])
        valid_df = bt_df.dropna(subset=['一個月內漲幅(%)'])
        total_count = len(valid_df)
        
        win_rate = int((win_count / total_count) * 100) if total_count > 0 else 0
        avg_ret = round(valid_df['一個月內漲幅(%)'].mean(), 2) if total_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("有效交易次數", total_count)
        col2.metric("20日後上漲機率", f"{win_rate}%")
        col3.metric("平均月漲幅", f"{avg_ret}%")
        
        show_cols = ['訊號日期', 'StockID', '名稱', '訊號價', '未來20日收盤', '一個月內漲幅(%)']
        def color_ret(val):
            if pd.isna(val): return ''
            color = 'red' if val > 0 else 'green'
            return f'color: {color}'
            
        st.dataframe(
            bt_df[show_cols].style.map(color_ret, subset=['一個月內漲幅(%)']), 
            use_container_width=True
        )
    else:
        st.warning("在此回測期間內，沒有股票符合您目前勾選的條件組合。")
    st.markdown("---")

# 3. 日常篩選
if st.session_state['master_df'] is not None:
    df = st.session_state['master_df'].copy()
    
    if '生命線' not in df.columns:
        st.error("⚠️ 資料結構已更新！請點擊左側紅色的 **「🔄 更新股價資料」** 按鈕。")
        st.stop()

    df = df[df['abs_bias'] <= bias_threshold]
    df = df[df['成交量'] >= (min_vol_input * 1000)]
    
    if filter_trend_up and filter_trend_down:
        st.error("❌ 請勿同時勾選「生命線向上」與「生命線向下」！")
        df = df[0:0] 
    elif filter_trend_up:
        df = df[df['生命線趨勢'] == "⬆️向上"]
    elif filter_trend_down:
        df = df[df['生命線趨勢'] == "⬇️向下"]

    if filter_treasure: df = df[df['浴火重生'] == True]
    if filter_kd: df = df[df['K值'] > df['D值']]
    
    if filter_vol_double: 
        df = df[df['成交量'] > (df['昨日成交量'] * 1.5)]

    # New: 籌碼篩選
    if filter_obv:
        df = df[df['OBV趨勢'] == "🔥吸籌"]
        
    if len(df) == 0:
        st.warning(f"⚠️ 找不到符合條件的股票！")
    else:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #ff4b4b;">
            <h2 style="color: #333; margin:0;">🔍 根據目前條件，共篩選出 <span style="color: #ff4b4b; font-size: 1.5em;">{len(df)}</span> 檔股票</h2>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        df['成交量(張)'] = (df['成交量'] / 1000).astype(int)
        df['KD值'] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
        df['選股標籤'] = df['代號'] + " " + df['名稱']
        
        display_cols = ['代號', '名稱', '收盤價', '生命線', '生命線趨勢', '乖離率(%)', '位置', 'KD值', '成交量(張)', 'OBV趨勢']
        if filter_treasure:
             df = df.sort_values(by='成交量', ascending=False)
        else:
             df = df.sort_values(by='abs_bias')
        
        tab1, tab2 = st.tabs(["📋 篩選結果列表", "📊 日趨勢圖"])
        
        with tab1:
            def highlight_row(row):
                if row['位置'] == "🟢生命線上":
                    return ['background-color: #e6fffa; color: black'] * len(row)
                else:
                    return ['background-color: #fff0f0; color: black'] * len(row)

            st.dataframe(
                df[display_cols].style.apply(highlight_row, axis=1),
                use_container_width=True,
                hide_index=True
            )

        with tab2:
            st.markdown("### 🔍 個股近半年趨勢圖")
            if len(df) > 0:
                selected_stock_label = st.selectbox("請選擇一檔股票：", df['選股標籤'].tolist())
                selected_row = df[df['選股標籤'] == selected_stock_label].iloc[0]
                target_ticker = selected_row['完整代號']
                target_name = selected_row['名稱']
                
                plot_stock_chart(target_ticker, target_name)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("目前股價", selected_row['收盤價'])
                col2.metric("生命線", selected_row['生命線'], delta=f"{selected_row['乖離率(%)']}%")
                col3.metric("KD指標", selected_row['KD值'])

else:
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 更新股價資料」** 按鈕開始挖寶！")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("welcome.jpg"):
            st.image("welcome.jpg", width=180)
        else:
            st.info("💡 歡迎使用旺來-台股生命線系統！")
