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
VER = "ver3.15 (Mobile UI)"
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
                # 處理 MultiIndex
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
                
                last_vol_series = df_v.iloc[-1]
                prev_vol_series = df_v.iloc[-2]

                recent_close_df = df_c.iloc[-8:]
                recent_ma200_df = ma200_df.iloc[-8:]

                for ticker in df_c.columns:
                    try:
                        price = last_price_series[ticker]
                        ma200 = last_ma200_series[ticker]
                        ma20 = last_ma20_series[ticker]
                        ma60 = last_ma60_series[ticker]
                        prev_ma200 = prev_ma200_series[ticker]
                        vol = last_vol_series[ticker]
                        prev_vol = prev_vol_series[ticker]
                        
                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0: continue

                        ma_trend = "⬆️" if ma200 >= prev_ma200 else "⬇️"

                        is_treasure = False
                        my_recent_c = recent_close_df[ticker]
                        my_recent_ma = recent_ma200_df[ticker]
                        if len(my_recent_c) >= 8:
                            cond_today_up = my_recent_c.iloc[-1] > my_recent_ma.iloc[-1]
                            past_c = my_recent_c.iloc[:-1]
                            past_ma = my_recent_ma.iloc[:-1]
                            cond_past_down = (past_c < past_ma).any()
                            if cond_today_up and cond_past_down: is_treasure = True

                        is_royal = False
                        if (price > ma20) and (ma20 > ma60) and (ma60 > ma200):
                            is_royal = True

                        stock_df = pd.DataFrame({'Close': df_c[ticker], 'High': df_h[ticker], 'Low': df_l[ticker]}).dropna()
                        k_val, d_val = 0, 0
                        if len(stock_df) >= 9:
                            k_val, d_val = calculate_kd_values(stock_df)

                        bias = ((price - ma200) / ma200) * 100
                        stock_info = stock_dict.get(ticker)
                        if not stock_info: continue
                        
                        code_pure = stock_info['code']
                        name_pure = stock_info['name']

                        raw_data_list.append({
                            '代號': code_pure,
                            '名稱': name_pure,
                            '代號與名稱': f"{code_pure} {name_pure}", # 手機好讀版
                            '完整代號': ticker,
                            '收盤價': float(price),
                            '生命線': float(ma200),
                            'MA20': float(ma20),
                            'MA60': float(ma60),
                            '趨勢': ma_trend,
                            '乖離率': float(bias),
                            'abs_bias': abs(float(bias)),
                            '成交量': int(vol),
                            '昨日成交量': int(prev_vol),
                            'K值': float(k_val),
                            'D值': float(d_val),
                            '位置': "🟢線上" if price >= ma200 else "🔴線下",
                            '浴火重生': is_treasure,
                            '皇冠特選': is_royal
                        })
                    except: continue
        except: pass
        
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"挖掘中...({int(current_progress*100)}%)")
        time.sleep(0.05)
    
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
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['60MA'] = df['Close'].rolling(window=60).mean()
        
        plot_df = df.tail(120).copy()
        plot_df['DateStr'] = plot_df.index.strftime('%Y-%m-%d')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['Close'], mode='lines', name='收盤價', line=dict(color='#00CC96', width=2.5)))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['20MA'], mode='lines', name='月線', line=dict(color='#AB63FA', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['60MA'], mode='lines', name='季線', line=dict(color='#19D3F3', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=plot_df['DateStr'], y=plot_df['200MA'], mode='lines', name='生命線', line=dict(color='#FFA15A', width=3)))

        fig.update_layout(
            title=f"📊 {name} ({ticker})", 
            yaxis_title='價格', 
            height=450, 
            margin=dict(l=20, r=20, t=50, b=20), # 減少邊距，手機更好看
            hovermode="x unified",
            xaxis=dict(type='category', tickangle=-45, nticks=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"繪圖失敗: {e}")

# --- 3. 介面顯示區 ---
st.title(f"🍍 {VER} 旺來-台股生命線")
st.markdown("---")

if 'master_df' not in st.session_state: st.session_state['master_df'] = None
if 'last_update' not in st.session_state: st.session_state['last_update'] = None
if 'backtest_result' not in st.session_state: st.session_state['backtest_result'] = None

with st.sidebar:
    st.header("資料庫")
    if st.button("🚨 重置系統"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    if st.button("🔄 更新資料 (開市用)", type="primary"):
        stock_dict = get_stock_list()
        if stock_dict:
            status = st.empty()
            prog = st.progress(0, text="準備中...")
            df = fetch_all_data(stock_dict, prog, status)
            st.session_state['master_df'] = df
            st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            prog.empty()
            st.success(f"更新共 {len(df)} 檔")
        
    if st.session_state['last_update']:
        st.caption(f"更新：{st.session_state['last_update']}")
    
    st.divider()
    st.header("篩選設定")
    bias_threshold = st.slider("乖離率 (±%)", 0.5, 5.0, 2.5, step=0.1)
    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    
    strategy_mode = st.radio("策略：", ("🛡️ 守護生命線", "🔥 浴火重生", "👑 皇冠特選"))

    filter_trend_up = False
    filter_trend_down = False
    filter_kd = False
    filter_vol_double = False
    filter_royal = False
    filter_treasure = False

    if strategy_mode == "🛡️ 守護生命線":
        c1, c2 = st.columns(2)
        with c1: filter_trend_up = st.checkbox("生命線向上", value=False)
        with c2: filter_trend_down = st.checkbox("生命線向下", value=False)
        filter_kd = st.checkbox("KD 黃金交叉", value=False)
        filter_vol_double = st.checkbox("爆量 (>1.5倍)", value=False)
    elif strategy_mode == "🔥 浴火重生":
        filter_treasure = True
        filter_vol_double = st.checkbox("爆量確認", value=False)
    elif strategy_mode == "👑 皇冠特選":
        filter_royal = True
        filter_vol_double = st.checkbox("爆量確認", value=False)

    st.divider()
    
    if st.button("🧪 策略回測"):
        st.info("阿吉正在調閱歷史檔案... ⏳")
        stock_dict = get_stock_list()
        prog = st.progress(0, text="初始化...")
        
        bt_df = run_strategy_backtest(
            stock_dict, prog, 
            use_trend_up=filter_trend_up, 
            use_treasure=filter_treasure, 
            use_vol=filter_vol_double,
            use_royal=filter_royal,
            min_vol_threshold=min_vol_input 
        )
        st.session_state['backtest_result'] = bt_df
        prog.empty()
        st.success("完成！")

# 主畫面 - 回測報告
if st.session_state['backtest_result'] is not None:
    bt_df = st.session_state['backtest_result']
    st.markdown("---")
    st.subheader(f"🧪 回測報告：{strategy_mode}")

    df_history = bt_df[bt_df['結果'] != "👀 觀察中"].copy()
    df_watching = bt_df[bt_df['結果'] == "👀 觀察中"].copy()
    
    # 手機版最佳化：定義欄位顯示規則 (使用 st.column_config)
    # 定義通用的 Column Config，讓表格變漂亮
    common_cols_config = {
        "最高漲幅": st.column_config.NumberColumn(
            "最高漲幅",
            help="訊號出現後的最高漲幅",
            format="%.2f %%",
            # 使用台股紅綠色，正數紅，負數綠
            step=0.01,
        ),
        "訊號價": st.column_config.NumberColumn("訊號價", format="$ %.2f"),
        "訊號日期": st.column_config.DateColumn("日期", format="MM-DD"),
        "結果": st.column_config.TextColumn("狀態"),
    }

    # 1. 關注中
    if not df_watching.empty:
        st.markdown(f"""
        <div style="background-color: #fff8dc; padding: 10px; border-radius: 10px; border-left: 5px solid #ffa500; margin-bottom: 20px;">
            <h4 style="color: #d2691e; margin:0;">👀 旺來關注中 (進行中)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        df_watching = df_watching.sort_values(by='訊號日期', ascending=False)
        
        # 視覺化漲跌幅 (台股紅漲綠跌)
        def color_surprise(val):
            color = '#ff4b4b' if val > 0 else '#00cc96'
            return f'color: {color}; font-weight: bold;'

        # 顯示精簡表格
        st.dataframe(
            df_watching[['代號與名稱', '訊號日期', '訊號價', '最高漲幅', '結果']].style.applymap(color_surprise, subset=['最高漲幅']),
            use_container_width=True, 
            hide_index=True,
            column_config=common_cols_config
        )
    else:
        st.info("👀 無關注中股票。")

    st.markdown("---")
    st.markdown("### 📜 歷史結算數據")

    if len(df_history) > 0:
        months = sorted(df_history['月份'].unique())
        tabs = st.tabs(["📊 總覽"] + months)
        
        with tabs[0]:
            win_df = df_history[df_history['結果'].str.contains("Win") | df_history['結果'].str.contains("驗證成功")]
            win_rate = int((len(win_df) / len(df_history)) * 100)
            avg_ret = round(df_history['最高漲幅'].mean(), 2)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("總次數", len(df_history))
            c2.metric("獲利機率", f"{win_rate}%")
            c3.metric("平均損益", f"{avg_ret}%")
            
            # 使用 Column Config 顯示 Bar Chart
            history_config = common_cols_config.copy()
            history_config["最高漲幅"] = st.column_config.NumberColumn(
                "最高漲幅", format="%.2f %%"
            )
            
            st.dataframe(
                df_history[['代號與名稱', '訊號日期', '訊號價', '最高漲幅', '結果']].style.applymap(lambda v: f'color: {"#ff4b4b" if v>0 else "#00cc96"}; font-weight: bold;', subset=['最高漲幅']),
                use_container_width=True,
                hide_index=True,
                column_config=history_config
            )

        for i, m in enumerate(months):
            with tabs[i+1]:
                m_df = df_history[df_history['月份'] == m]
                # 這裡簡單顯示，手機版不宜過度複雜
                st.dataframe(
                     m_df[['代號與名稱', '訊號日期', '訊號價', '最高漲幅', '結果']].style.applymap(lambda v: f'color: {"#ff4b4b" if v>0 else "#00cc96"}; font-weight: bold;', subset=['最高漲幅']),
                     use_container_width=True,
                     hide_index=True,
                     column_config=history_config
                )
    else:
        st.warning("無歷史符合條件股票。")
    st.markdown("---")

# 主畫面 - 日常篩選
if st.session_state['master_df'] is not None:
    df = st.session_state['master_df'].copy()
    
    if '生命線' not in df.columns:
        st.error("⚠️ 請重新更新資料！")
        st.stop()

    df = df[df['成交量'] >= (min_vol_input * 1000)]
    
    if strategy_mode == "🔥 浴火重生":
        df = df[df['浴火重生'] == True]
    elif strategy_mode == "👑 皇冠特選":
        if '皇冠特選' in df.columns: df = df[df['皇冠特選'] == True]
        else: df = df[(df['收盤價'] > df['MA20']) & (df['MA20'] > df['MA60']) & (df['MA60'] > df['生命線'])]
    else:
        df = df[df['abs_bias'] <= bias_threshold]
        if filter_trend_up: df = df[df['趨勢'] == "⬆️"]
        elif filter_trend_down: df = df[df['趨勢'] == "⬇️"]
        if filter_kd: df = df[df['K值'] > df['D值']]
    
    if filter_vol_double: 
        df = df[df['成交量'] > (df['昨日成交量'] * 1.5)]
        
    if len(df) == 0:
        st.warning(f"⚠️ 找不到股票！")
    else:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; text-align: center; border: 2px solid #ff4b4b;">
            <h3 style="color: #333; margin:0;">🔍 篩選出 <span style="color: #ff4b4b;">{len(df)}</span> 檔</h3>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        df['成交量(張)'] = (df['成交量'] / 1000).astype(int)
        df['KD值'] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
        
        # 定義顯示欄位 (手機最佳化: 合併代號名稱)
        display_cols = ['代號與名稱', '收盤價', '生命線', '乖離率', '位置', 'KD值', '成交量(張)']
        if strategy_mode == "👑 皇冠特選":
            display_cols = ['代號與名稱', '收盤價', 'MA20', 'MA60', 'KD值', '成交量(張)']
            
        df = df.sort_values(by='成交量', ascending=False)
        
        tab1, tab2 = st.tabs(["📋 列表", "📊 圖表"])
        
        with tab1:
            # 使用 column_config 視覺化
            st.dataframe(
                df[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "收盤價": st.column_config.NumberColumn("收盤價", format="$%.2f"),
                    "生命線": st.column_config.NumberColumn("生命線", format="$%.2f"),
                    "乖離率": st.column_config.NumberColumn(
                        "乖離率", 
                        format="%.2f%%",
                        help="距離生命線的百分比",
                    ),
                    "成交量(張)": st.column_config.ProgressColumn(
                        "成交量",
                        format="%d 張",
                        min_value=0,
                        max_value=int(df['成交量(張)'].max()),
                    ),
                }
            )

        with tab2:
            st.markdown("### 🔍 個股趨勢")
            if len(df) > 0:
                # 這裡的 key 改用代號與名稱，比較好選
                selected_stock_label = st.selectbox("選擇股票：", df['代號與名稱'].tolist())
                selected_row = df[df['代號與名稱'] == selected_stock_label].iloc[0]
                plot_stock_chart(selected_row['完整代號'], selected_row['名稱'])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("收盤", f"{selected_row['收盤價']:.2f}")
                c2.metric("量", f"{selected_row['成交量(張)']} 張")
                c3.metric("KD", selected_row['KD值'])

else:
    st.warning("👈 請先點擊左側 **「🔄 更新資料」**")
    if os.path.exists("welcome.jpg"):
        c1, c2, c3 = st.columns([1,2,1])
        with c2: st.image("welcome.jpg", width=150)
