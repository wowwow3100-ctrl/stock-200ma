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
VER = "ver4.2_Ultimate"
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

# --- 【核心修正】策略回測函數 (整合第二版邏輯) ---
def run_strategy_backtest(stock_dict, progress_bar, use_trend_up, use_treasure, use_vol):
    results = []
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    for i, batch_idx in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch = all_tickers[batch_idx : batch_idx + BATCH_SIZE]
        try:
            # 下載 2 年數據以確保有足夠的移動平均線和未來驗證數據
            data = yf.download(batch, period="2y", interval="1d", progress=False, auto_adjust=False)
            if not data.empty:
                try:
                    df_c = data['Close']
                    df_v = data['Volume']
                    df_l = data['Low']
                    df_h = data['High']
                except KeyError:
                    continue
                
                # 處理單一股票的情況 (Series 轉 DataFrame)
                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])

                ma200_df = df_c.rolling(window=200).mean()
                
                # 掃描窗口：保留最後 20 天作為驗證期 (避免 index out of bound)，只回測到 20 天前
                scan_window = df_c.index[-250:-20] 
                
                for ticker in df_c.columns:
                    try:
                        c_series = df_c[ticker]
                        v_series = df_v[ticker]
                        l_series = df_l[ticker]
                        ma_series = ma200_df[ticker]
                        
                        stock_name = stock_dict.get(ticker, {}).get('name', ticker)
                        stock_code = stock_dict.get(ticker, {}).get('code', ticker.split('.')[0])
                        
                        for date in scan_window:
                            if pd.isna(ma_series[date]): continue
                            
                            idx = c_series.index.get_loc(date)
                            # 確保有足夠的歷史數據進行判斷
                            if idx < 20: continue 

                            close_p = c_series.iloc[idx]
                            low_p = l_series.iloc[idx]
                            vol = v_series.iloc[idx]
                            prev_vol = v_series.iloc[idx-1]
                            ma_val = ma_series.iloc[idx]
                            ma_val_20ago = ma_series.iloc[idx-20]
                            
                            if ma_val == 0 or prev_vol == 0: continue

                            is_match = False
                            
                            # --- 策略判斷邏輯 (與第一版相同) ---
                            if use_trend_up and (ma_val <= ma_val_20ago): continue
                            if use_vol and (vol <= prev_vol * 1.5): continue

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
                                # 一般策略：接近均線且在均線上
                                cond_near = (low_p <= ma_val * 1.03) and (low_p >= ma_val * 0.90) 
                                cond_up = (close_p > ma_val)
                                if cond_near and cond_up: is_match = True
                            
                            # --- 【第二版邏輯植入】驗證與數據計算 ---
                            if is_match:
                                # 1. 驗證：抓取未來第 20 個交易日的收盤價
                                # 檢查是否還有未來 20 天的數據
                                if idx + 20 < len(c_series):
                                    future_close = c_series.iloc[idx + 20]
                                    profit_pct = (future_close - close_p) / close_p * 100
                                    
                                    if profit_pct > 0:
                                        result_status = "Win (上漲)"
                                    else:
                                        result_status = "Loss (下跌)"
                                else:
                                    # 如果是最近一個月內的訊號，還沒有第 20 天的數據
                                    profit_pct = np.nan
                                    result_status = "統計中"

                                month_str = date.strftime('%Y-%m') # 使用 年-月 格式方便排序
                                
                                results.append({
                                    '月份': month_str,
                                    'StockID': stock_code, # 為了配合第二版邏輯，使用 StockID
                                    '名稱': stock_name,
                                    'Date': date, # 保留 datetime 物件方便排序
                                    '訊號日期': date.strftime('%Y-%m-%d'),
                                    '訊號價': float(close_p),
                                    '未來20日收盤': float(future_close) if not np.isnan(profit_pct) else np.nan,
                                    '一個月內漲幅(%)': float(profit_pct) if not np.isnan(profit_pct) else np.nan,
                                    '結果': result_status
                                })
                                # 一個月內同一支股票只取一次訊號，避免重複計算 (Skip next 20 days)
                                # 這裡簡化處理，直接 break 當月循環或由使用者自行判斷
                                # 在此版本我們記錄所有觸發點，讓「觸發次數」功能生效
                                
                    except Exception as e:
                        continue
        except:
            pass
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中 (整合第二版驗證邏輯)...({int(progress*100)}%)")
        
    # --- 【第二版後處理】統計與日誌生成 ---
    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # 1. 觸發次數統計 (Count)
    df_results['觸發次數'] = df_results.groupby('StockID')['StockID'].transform('count')

    # 2. 數據淨化 (Rounding)
    numeric_cols = ['訊號價', '未來20日收盤', '一個月內漲幅(%)']
    for col in numeric_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(2)

    # 3. 詳細日誌 (Log)
    def generate_log(row):
        rise_pct = f"{row['一個月內漲幅(%)']}%" if not pd.isna(row['一個月內漲幅(%)']) else "統計中"
        return (f"日期: {row['訊號日期']} | "
                f"股票: {row['StockID']} | "
                f"觸發價: {row['訊號價']} | "
                f"累計觸發: {row['觸發次數']}次 | "
                f"後續漲幅: {rise_pct}")

    df_results['紀錄日誌'] = df_results.apply(generate_log, axis=1)
    
    # 依照日期排序
    df_results = df_results.sort_values(by=['Date', 'StockID'], ascending=[False, True])
    
    return df_results

# --- 即時資料抓取 (維持第一版架構，加入數據淨化) ---
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
                            '收盤價': round(float(price), 2),     # 數據淨化
                            '生命線': round(float(ma200), 2),     # 數據淨化
                            '生命線趨勢': ma_trend,
                            '乖離率(%)': round(float(bias), 2),   # 數據淨化
                            'abs_bias': abs(float(bias)),
                            '成交量': int(vol),
                            '昨日成交量': int(prev_vol),
                            'K值': round(float(k_val), 2),        # 數據淨化
                            'D值': round(float(d_val), 2),        # 數據淨化
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
        
        # 純線圖 (Line Chart) - 第一版風格
        fig.add_trace(go.Scatter(
            x=plot_df['DateStr'], 
            y=plot_df['Close'], 
            mode='lines',
            name='收盤價',
            line=dict(color='#00CC96', width=2.5) 
        ))
        
        fig.add_trace(go.Scatter(
            x=plot_df['DateStr'], 
            y=plot_df['200MA'], 
            mode='lines',
            name='生命線',
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
    filter_trend_down = st.checkbox("📉 生命線向下 (空方壓力)", value=False)
    
    filter_treasure = st.checkbox("🔥 浴火重生 (假跌破拉回)", value=False)
    st.caption("🔍 尋找過去7日內曾跌破，但今日站回生命線的強勢股")
    
    filter_kd = st.checkbox("KD 黃金交叉 (K > D)", value=False)
    filter_vol_double = st.checkbox("出量 (今日 > 昨日x1.5)", value=False)
    
    st.divider()
    
    st.caption("⚠️ 注意：回測需調閱2年歷史資料，運算時間較長。")
    if st.button("🧪 策略回測 (含20日後漲幅驗證)"):
        st.info("阿吉正在調閱歷史檔案，進行第二版邏輯驗證... (請稍候) ⏳")
        stock_dict = get_stock_list()
        bt_progress = st.progress(0, text="初始化回測...")
        
        bt_df = run_strategy_backtest(
            stock_dict, 
            bt_progress, 
            use_trend_up=filter_trend_up, 
            use_treasure=filter_treasure, 
            use_vol=filter_vol_double
        )
        
        st.session_state['backtest_result'] = bt_df
        bt_progress.empty()
        st.success("回測完成！已生成詳細報表。")

    with st.expander("📅 系統開發日誌"):
        st.markdown("""
        ### Ver 4.2 (Hybrid)
        * **Merge**: 完美結合第一版介面與第二版驗證核心。
        * **Logic**: 驗證指標改為「訊號後第20日收盤價」計算真實月漲幅。
        * **Feature**: 新增「觸發次數」統計，識別熱門股。
        * **UI**: 報表增加「紀錄日誌」字串，數據全面保留小數點後兩位。
        """)

# 主畫面 - 回測報告
if st.session_state['backtest_result'] is not None:
    bt_df = st.session_state['backtest_result']
    st.markdown("---")
    
    strategy_name = "基礎策略"
    if filter_treasure: strategy_name = "浴火重生(假跌破)"
    elif filter_trend_up: strategy_name = "趨勢向上 + 支撐"
    
    st.subheader(f"🧪 策略回測報告：{strategy_name}")
    st.caption("驗證邏輯：計算訊號觸發後，持有 **20個交易日(約一個月)** 的漲跌幅表現。")
    
    if len(bt_df) > 0:
        months = sorted(bt_df['月份'].unique(), reverse=True) # 新的月份在前面
        
        tabs = st.tabs(["📊 總覽 (含日誌)"] + months)
        
        with tabs[0]:
            # 統計
            win_count = len(bt_df[bt_df['結果'].str.contains("Win")])
            valid_df = bt_df.dropna(subset=['一個月內漲幅(%)']) # 只計算有驗證結果的
            total_count = len(valid_df)
            
            win_rate = int((win_count / total_count) * 100) if total_count > 0 else 0
            avg_ret = round(valid_df['一個月內漲幅(%)'].mean(), 2) if total_count > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("有效驗證次數", total_count)
            col2.metric("20日後上漲機率", f"{win_rate}%")
            col3.metric("平均月漲幅", f"{avg_ret}%")
            
            # 顯示完整表格 (含新欄位)
            show_cols = ['訊號日期', 'StockID', '名稱', '訊號價', '觸發次數', '未來20日收盤', '一個月內漲幅(%)', '紀錄日誌']
            
            def color_ret(val):
                if pd.isna(val): return ''
                color = 'red' if val > 0 else 'green'
                return f'color: {color}'
                
            st.dataframe(
                bt_df[show_cols].style.map(color_ret, subset=['一個月內漲幅(%)']), 
                use_container_width=True
            )

        for i, m in enumerate(months):
            with tabs[i+1]:
                m_df = bt_df[bt_df['月份'] == m]
                
                m_valid = m_df.dropna(subset=['一個月內漲幅(%)'])
                m_win = len(m_valid[m_valid['一個月內漲幅(%)'] > 0])
                m_total = len(m_valid)
                m_rate = int((m_win / m_total) * 100) if m_total > 0 else 0
    
