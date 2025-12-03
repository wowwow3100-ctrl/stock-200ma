import streamlit as st
import yfinance as yf
import pandas as pd
import twstock
import time
from datetime import datetime
import plotly.graph_objects as go
import requests

# --- 1. 網頁設定 ---
VER = "ver2.5"
st.set_page_config(page_title=f"🍍 旺來-台股生命線({VER})", layout="wide")

# --- 2. 核心功能區 ---
@st.cache_data(ttl=3600)
def get_stock_list():
    """取得台股清單 (排除金融/ETF)"""
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

def analyze_backtest(df):
    """
    回測邏輯：
    1. 針對該股票過去 30 天的歷史數據。
    2. 找出「最低價碰到生命線」且「收盤價守住生命線」的日子 (Touch Event)。
    3. 檢查該日子後的 5 天內，股價是否上漲 (收盤價 > 觸碰日收盤價)。
    回傳: (總觸碰次數, 成功反彈次數)
    """
    try:
        # 確保有足夠數據算 200MA
        if len(df) < 250: return 0, 0
        
        # 計算 200MA
        ma200 = df['Close'].rolling(window=200).mean()
        
        # 取最近 30 天 (保留最後 5 天做驗證，所以只檢查 day -30 到 day -5)
        # 這樣才能確認「之後」有沒有漲
        check_window = df.iloc[-35:-5]
        
        touch_count = 0
        win_count = 0
        
        for i in range(len(check_window)):
            date = check_window.index[i]
            price_low = check_window['Low'].iloc[i]
            price_close = check_window['Close'].iloc[i]
            ma_val = ma200.loc[date]
            
            if pd.isna(ma_val): continue
            
            # 條件：最低價跌破或碰到生命線 (1%緩衝)，但收盤價站穩 (或在線下 1% 以內)
            # 這裡定義寬鬆一點：只要 Low <= MA * 1.01 就算碰到
            if price_low <= ma_val * 1.01:
                touch_count += 1
                
                # 檢查後續 5 天的表現
                # 取得該日之後的 5 天數據
                future_idx = df.index.get_loc(date)
                future_prices = df['Close'].iloc[future_idx+1 : future_idx+6]
                
                if len(future_prices) > 0:
                    max_future = future_prices.max()
                    # 如果後續 5 天內最高價 > 觸碰日收盤價 * 1.02 (漲2%)
                    if max_future > price_close * 1.02:
                        win_count += 1
                        
        return touch_count, win_count
    except:
        return 0, 0

def fetch_all_data(stock_dict, progress_bar, status_text):
    all_tickers = list(stock_dict.keys())
    BATCH_SIZE = 30
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    raw_data_list = []

    # 全局回測統計
    global_touches = 0
    global_wins = 0

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
                last_vol_series = df_v.iloc[-1]
                prev_vol_series = df_v.iloc[-2]

                recent_close_df = df_c.iloc[-8:]
                recent_ma200_df = ma200_df.iloc[-8:]

                for ticker in df_c.columns:
                    try:
                        price = last_price_series[ticker]
                        ma200 = last_ma200_series[ticker]
                        vol = last_vol_series[ticker]
                        prev_vol = prev_vol_series[ticker]
                        
                        if pd.isna(price) or pd.isna(ma200) or ma200 == 0: continue

                        # 1. 開寶箱判定
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

                        # 2. 執行個股回測 (近一個月)
                        stock_df = pd.DataFrame({
                            'Close': df_c[ticker], 'High': df_h[ticker], 'Low': df_l[ticker]
                        }).dropna()
                        
                        t_count, w_count = analyze_backtest(stock_df)
                        global_touches += t_count
                        global_wins += w_count
                        
                        # 3. KD 計算
                        k_val, d_val = 0, 0
                        if len(stock_df) >= 9:
                            k_val, d_val = calculate_kd_values(stock_df)

                        bias = ((price - ma200) / ma200) * 100
                        stock_info = stock_dict.get(ticker)
                        if not stock_info: continue
                        
                        # 4. 整理回測數據字串
                        backtest_str = "無"
                        if t_count > 0:
                            win_rate = int((w_count / t_count) * 100)
                            backtest_str = f"{win_rate}% ({w_count}/{t_count})"

                        raw_data_list.append({
                            '代號': stock_info['code'],
                            '名稱': stock_info['name'],
                            '收盤價': float(price),
                            '生命線(200MA)': float(ma200),
                            '乖離率(%)': float(bias),
                            'abs_bias': abs(float(bias)),
                            '成交量': int(vol),
                            '昨日成交量': int(prev_vol),
                            'K值': float(k_val),
                            'D值': float(d_val),
                            '位置': "🟢生命線上" if price >= ma200 else "🔴生命線下",
                            '開寶箱': is_treasure,
                            '近月反彈勝率': backtest_str # 新增欄位
                        })
                    except: continue
        except: pass
        
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"正在開鎖寶箱...({int(current_progress*100)}%)")
        time.sleep(0.05)
    
    # 將全局回測結果存入 Session State 供介面使用
    if global_touches > 0:
        global_win_rate = int((global_wins / global_touches) * 100)
    else:
        global_win_rate = 0
    st.session_state['global_backtest'] = {
        'touches': global_touches,
        'wins': global_wins,
        'rate': global_win_rate
    }
    
    return pd.DataFrame(raw_data_list)

# --- 3. 介面顯示區 ---
st.title(f"🍍 {VER} 旺來-台股生命線")
st.markdown("---")

if 'master_df' not in st.session_state:
    st.session_state['master_df'] = None
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None
if 'global_backtest' not in st.session_state:
    st.session_state['global_backtest'] = None

with st.sidebar:
    st.header("1. 資料庫管理")
    
    if st.button("🔄 更新股價資料 (開市請按我)", type="primary"):
        stock_dict = get_stock_list()
        
        # Emoji 動畫
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
    filter_treasure = st.checkbox("🎁 開寶箱 (假跌破生命線)", value=False)
    st.caption("🔍 尋找過去7日內曾跌破，但今日站回生命線的強勢股")
    filter_kd = st.checkbox("KD 黃金交叉 (K > D)", value=False)
    filter_vol_double = st.checkbox("爆量 (今日 > 昨日x2)", value=False)
    filter_ma_up = st.checkbox("只看站上生命線 (多方)", value=False)
    
    st.divider()
    # --- 新增功能：策略驗證按鈕 ---
    show_backtest = st.checkbox("🧪 顯示近一月策略勝率", value=False)
    
    st.divider()
    with st.expander("📅 版本開發紀錄"):
        st.markdown("""
        **Ver 2.5 (Strategy Backtest)**
        - 新增：策略驗證功能。統計過去一個月所有觸碰生命線股票的反彈勝率。
        - 介面：移除 K 線圖，改為純數據表格與驗證報告。
        - 視覺：更新歡迎畫面為「寶箱炸開」GIF。
        """)

# 主畫面
if st.session_state['master_df'] is not None:
    df = st.session_state['master_df'].copy()
    
    # --- 顯示策略驗證看板 (如果勾選) ---
    if show_backtest and st.session_state['global_backtest']:
        bt = st.session_state['global_backtest']
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #00a8cc; margin-bottom: 20px;">
            <h3 style="margin:0; color: #00607a;">🧪 生命線戰法 - 近月準確度驗證</h3>
            <p>在過去 30 天內，全台股共有 <b>{bt['touches']}</b> 次觸碰生命線紀錄。</p>
            <p>其中有 <b>{bt['wins']}</b> 次在隨後 5 日內成功反彈 (漲幅 > 2%)。</p>
            <h2 style="color: #00a8cc; margin:0;">🔥 近期勝率：{bt['rate']}%</h2>
        </div>
        """, unsafe_allow_html=True)

    # 篩選邏輯
    df = df[df['abs_bias'] <= bias_threshold]
    df = df[df['成交量'] >= (min_vol_input * 1000)]
    
    if filter_treasure: df = df[df['開寶箱'] == True]
    if filter_kd: df = df[df['K值'] > df['D值']]
    if filter_vol_double: df = df[df['成交量'] > (df['昨日成交量'] * 2)]
    if filter_ma_up: df = df[df['位置'] == "🟢生命線上"]

    if len(df) == 0:
        st.warning(f"⚠️ 找不到符合條件的股票！\n\n請嘗試放寬乖離率範圍 (例如拉大到 5%) 或是取消部分勾選。")
    else:
        # 標題看板
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #ff4b4b;">
            <h2 style="color: #333; margin:0;">🔍 根據目前條件，共篩選出 <span style="color: #ff4b4b; font-size: 1.5em;">{len(df)}</span> 檔股票</h2>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        df['成交量(張)'] = (df['成交量'] / 1000).astype(int)
        df['KD值'] = df.apply(lambda x: f"K:{int(x['K值'])} D:{int(x['D值'])}", axis=1)
        
        # 顯示欄位：移除圖表，加入回測勝率
        display_cols = ['代號', '名稱', '收盤價', '生命線(200MA)', '乖離率(%)', '成交量(張)', '位置', 'KD值', '近月反彈勝率']
        
        if filter_treasure:
             df = df.sort_values(by='成交量', ascending=False)
        else:
             df = df.sort_values(by='abs_bias')
        
        # --- 顯示結果表格 (無圖表模式) ---
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

else:
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 更新股價資料」** 按鈕開始挖寶！")
    
    # --- 歡迎畫面：寶箱炸開 (符合您的要求) ---
    # 這裡放一個寶箱金幣的 GIF
    chest_explode_url = "https://cdn.pixabay.com/animation/2023/02/09/21/29/chest-7779776_512.gif"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(chest_explode_url, caption="💰 準備好了嗎？點擊左上角開始挖寶！")
