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
VER = "ver4.0"
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

# --- 策略回測核心函數 (動態連動版) ---
def run_strategy_backtest(stock_dict, progress_bar, use_trend_up, use_treasure, use_vol):
    """
    動態回測：根據使用者勾選的條件進行歷史驗證
    params:
        use_trend_up: 是否限制生命線向上
        use_treasure: 是否使用挖寶(假跌破)邏輯
        use_vol: 是否限制出量
    """
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
                scan_window = df_c.index[-90:-10] # 掃描近3個月
                
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

                            # --- 動態策略條件 ---
                            is_match = False
                            
                            # 1. 趨勢濾網 (如果勾選)
                            if use_trend_up and (ma_val <= ma_val_20ago):
                                continue # 趨勢不對就跳過

                            # 2. 出量濾網 (如果勾選)
                            if use_vol and (vol <= prev_vol * 1.5):
                                continue # 量不夠就跳過

                            # 3. 型態判斷 (挖寶 vs 一般站上)
                            if use_treasure:
                                # 挖寶邏輯：過去7天有跌破，今天站上
                                # 取得過去 8 天資料 (含今天)
                                start_idx = idx - 7
                                if start_idx < 0: continue
                                recent_c = c_series.iloc[start_idx : idx+1]
                                recent_ma = ma_series.iloc[start_idx : idx+1]
                                
                                # 今天站上
                                cond_today_up = recent_c.iloc[-1] > recent_ma.iloc[-1]
                                # 過去7天(不含今天)有跌破
                                past_c = recent_c.iloc[:-1]
                                past_ma = recent_ma.iloc[:-1]
                                cond_past_down = (past_c < past_ma).any()
                                
                                if cond_today_up and cond_past_down:
                                    is_match = True
                            else:
                                # 一般邏輯：接近生命線 + 站上
                                cond_near = (low_p <= ma_val * 1.03) and (low_p >= ma_val * 0.90) 
                                cond_up = (close_p > ma_val)
                                if cond_near and cond_up:
                                    is_match = True
                            
                            if is_match:
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
        progress_bar.progress(progress, text=f"深度回測中 (條件連動分析)...({int(progress*100)}%)")
        
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
    filter_trend_down = st.checkbox("📉 生命線向下 (空方壓力)", value=False)
    
    filter_treasure = st.checkbox("🔥 浴火重生 (假跌破拉回)", value=False)
    st.caption("🔍 尋找過去7日內曾跌破，但今日站回生命線的強勢股")
    
    filter_kd = st.checkbox("KD 黃金交叉 (K > D)", value=False)
    filter_vol_double = st.checkbox("出量 (今日 > 昨日x1.5)", value=False)
    
    st.divider()
    
    st.caption("⚠️ 注意：回測需調閱2年歷史資料，運算時間較長 (約2分鐘)。")
    # --- 關鍵修改：將按鈕變數傳入回測函數 ---
    if st.button("🧪 策略回測 (近3個月表現)"):
        st.info("阿吉正在調閱過去2年的歷史檔案，進行深度驗證... (請稍候) ⏳")
        stock_dict = get_stock_list()
        bt_progress = st.progress(0, text="初始化回測...")
        
        # 傳遞使用者目前的勾選狀態給回測系統
        bt_df = run_strategy_backtest(
            stock_dict, 
            bt_progress, 
            use_trend_up=filter_trend_up, 
            use_treasure=filter_treasure, 
            use_vol=filter_vol_double
        )
        
        st.session_state['backtest_result'] = bt_df
        bt_progress.empty()
        st.success("回測完成！請查看下方結果。")

    with st.expander("📅 系統開發日誌 (Changelog)"):
        st.markdown("""
        ### Ver 4.0 (Dynamic Backtest)
        * **Feature**: 策略回測現在會**連動**側邊欄的篩選條件。
            * 勾選「生命線向上」➜ 回測時過濾下降趨勢。
            * 勾選「浴火重生」➜ 回測時改抓假跌破型態。
            * 勾選「出量」➜ 回測時加入量能濾網。

        ### Ver 3.9 (Visual Polish)
        * **UI/UX**: 歡迎畫面美化，字體加深，圖片置中。
        """)

# 主畫面 - 回測報告
if st.session_state['backtest_result'] is not None:
    bt_df = st.session_state['backtest_result']
    st.markdown("---")
    
    # 根據勾選狀態顯示動態標題
    strategy_name = "基礎策略"
    if filter_treasure: strategy_name = "浴火重生(假跌破)"
    elif filter_trend_up: strategy_name = "趨勢向上 + 支撐"
    
    st.subheader(f"🧪 策略回測報告：{strategy_name} (歷史訊號驗證)")
    
    if len(bt_df) > 0:
        months = sorted(bt_df['月份'].unique())
        
        tabs = st.tabs(["📊 總覽"] + months)
        
        with tabs[0]:
            win_count = len(bt_df[bt_df['結果'].str.contains("Win")])
            total_count = len(bt_df)
            win_rate = int((win_count / total_count) * 100)
            avg_max_ret = round(bt_df['最高漲幅(%)'].mean(), 2)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("總觸發次數", total_count)
            col2.metric("總平均勝率 (反彈>3%)", f"{win_rate}%")
            col3.metric("總平均最高漲幅", f"{avg_max_ret}%")
            st.dataframe(bt_df, use_container_width=True)

        for i, m in enumerate(months):
            with tabs[i+1]:
                m_df = bt_df[bt_df['月份'] == m]
                
                m_win = len(m_df[m_df['結果'].str.contains("Win")])
                m_total = len(m_df)
                m_rate = int((m_win / m_total) * 100) if m_total > 0 else 0
                m_avg = round(m_df['最高漲幅(%)'].mean(), 2) if m_total > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{m} 觸發次數", m_total)
                c2.metric(f"{m} 勝率", f"{m_rate}%")
                c3.metric(f"{m} 平均漲幅", f"{m_avg}%")
                
                def color_ret(val):
                    color = 'red' if val > 0 else 'green'
                    return f'color: {color}'
                st.dataframe(m_df.style.map(color_ret, subset=['最高漲幅(%)']), use_container_width=True)

    else:
        st.warning("在此回測期間內，沒有股票符合您目前勾選的條件組合。")
    st.markdown("---")

# 主畫面 - 日常篩選
if st.session_state['master_df'] is not None:
    df = st.session_state['master_df'].copy()
    
    # 防呆
    if '生命線' not in df.columns:
        st.error("⚠️ 資料結構已更新！請點擊左側紅色的 **「🔄 更新股價資料」** 按鈕。")
        st.stop()

    df = df[df['abs_bias'] <= bias_threshold]
    df = df[df['成交量'] >= (min_vol_input * 1000)]
    
    if filter_trend_up and filter_trend_down:
        st.error("❌ 請勿同時勾選「生命線向上」與「生命線向下」，這兩個條件是互斥的！")
        df = df[0:0] 
    elif filter_trend_up:
        df = df[df['生命線趨勢'] == "⬆️向上"]
    elif filter_trend_down:
        df = df[df['生命線趨勢'] == "⬇️向下"]

    if filter_treasure: df = df[df['浴火重生'] == True]
    if filter_kd: df = df[df['K值'] > df['D值']]
    
    if filter_vol_double: 
        df = df[df['成交量'] > (df['昨日成交量'] * 1.5)]
        
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
        
        display_cols = ['代號', '名稱', '收盤價', '生命線', '生命線趨勢', '乖離率(%)', '位置', 'KD值', '成交量(張)']
        if filter_treasure:
             df = df.sort_values(by='成交量', ascending=False)
        else:
             df = df.sort_values(by='abs_bias')
        
        tab1, tab2 = st.tabs(["📋 篩選結果列表", "📊 日K線技術分析"])
        
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
            st.markdown("### 🔍 個股近半年日K線圖")
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
            st.markdown(
                """
                <div style="text-align: center; color: #333333; font-size: 1.1em; margin-bottom: 20px; line-height: 1.6; font-weight: bold;">
                    這是數年來的經驗收納<br>
                    此工具僅供參考，不代表投資建議<br>
                    預祝心想事成，從從容容，紫氣東來! 🟣✨
                </div>
                """,
                unsafe_allow_html=True
            )
            sub_c1, sub_c2, sub_c3 = st.columns([1, 1, 1])
            with sub_c2:
                 st.image("welcome.jpg", width=180)
        else:
            st.info("💡 尚未偵測到 welcome.jpg，請將您的紫色招財圖上傳至 GitHub 並命名為 welcome.jpg，這裡就會顯示囉！")
