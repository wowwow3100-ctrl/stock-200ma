這是一個很棒的反饋！這代表我們的「加速引擎」開太快，導致 Yahoo Finance 的伺服器啟動了防禦機制（Rate Limiting），把我們後半段的請求擋掉了，或者是一次請求太長導致數據遺失。

**原因分析：**

1.  **批次太大 (150檔)**：雖然理論上可以更快，但 Yahoo 伺服器有時候會拒絕太長的請求字串，或是回傳缺漏的數據。
2.  **頻率太快**：原本設定 `sleep(0.05)` (0.05秒)，這對伺服器來說幾乎是連續攻擊，容易被暫時封鎖 IP。

-----

### 🛠️ 修正方案 (Ver 3.18 Stability Fix)

我們將策略調整為 **「穩健加速」**：

1.  **Batch Size 調回 50**：這是 `yfinance` 社群公認最穩定的數字，雖然請求次數變多，但保證能拿到資料。
2.  **增加休息時間**：將間隔拉長到 `0.3` 秒，確保不被判定為惡意機器人。

請使用下方修正後的完整程式碼（我已將參數調校至最佳平衡點）：

```python
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
VER = "ver3.18 (Stability Fix)"
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
def run_strategy_backtest(stock_dict, progress_bar, use_trend_up, use_treasure, use_vol, use_royal, min_vol_threshold):
    results = []
    all_tickers = list(stock_dict.keys())
    
    # --- 修正: 回測時也使用穩定的 50 檔批次 ---
    BATCH_SIZE = 50 
    total_batches = (len(all_tickers) // BATCH_SIZE) + 1
    
    OBSERVE_DAYS = 20 if use_royal else 10
    
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
                                    cond_prev_down = recent_c.iloc[-2] <= recent_ma.iloc[-2]
                                    
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
                                            result_status = "Win (止盈出場) 👑"
                                            is_watching = False 
                                            break
                                        
                                        if day_close < day_ma200:
                                            final_profit_pct = (day_close - close_p) / close_p * 100
                                            result_status = "Loss (破線停損) 🛑"
                                            is_watching = False 
                                            break
                                    
                                    if is_watching:
                                        if days_after_signal >= OBSERVE_DAYS:
                                            end_close = c_series.iloc[idx + OBSERVE_DAYS]
                                            final_profit_pct = (end_close - close_p) / close_p * 100
                                            if final_profit_pct > 0: result_status = "Win (期滿獲利)"
                                            else: result_status = "Loss (期滿虧損)"
                                            is_watching = False
                                        else:
                                            result_status = "觀察中"

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
                                    '月份': '👀 關注中' if is_watching else month_str,
                                    '代號': ticker.replace(".TW", "").replace(".TWO", ""),
                                    '名稱': stock_name,
                                    '訊號日期': date.strftime('%Y-%m-%d'),
                                    '訊號價': round(close_p, 2),
                                    '最高漲幅(%)': round(final_profit_pct, 2),
                                    '結果': "觀察中" if is_watching else result_status
                                })
                                
                                if use_royal: 
                                    break 
                                
                    except:
                        continue
        except:
            pass
        
        progress = (i + 1) / total_batches
        progress_bar.progress(progress, text=f"深度回測中 (計算分月數據)...({int(progress*100)}%)")
        
    if not results:
        return pd.DataFrame(columns=['月份', '代號', '名稱', '訊號日期', '訊號價', '最高漲幅(%)', '結果'])

    return pd.DataFrame(results)

def fetch_all_data(stock_dict, progress_bar, status_text):
    if not stock_dict: return pd.DataFrame()
    
    all_tickers = list(stock_dict.keys())
    
    # --- 關鍵修正：將 Batch Size 調回 50，避免資料遺失 ---
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
                    df_v = data['Volume']
                except KeyError:
                    continue

                if isinstance(df_c, pd.Series):
                    df_c = df_c.to_frame(name=batch[0])
                    df_h = df_h.to_frame(name=batch[0])
                    df_l = df_l.to_frame(name=batch[0])
                    df_v = df_v.to_frame(name=batch[0])

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

                        raw_data_list.append({
                            '代號': stock_info['code'],
                            '名稱': stock_info['name'],
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
                            '皇冠特選': is_royal
                        })
                    except: continue
        except: pass
        
        current_progress = (i + 1) / total_batches
        progress_bar.progress(current_progress, text=f"系統正在努力挖掘寶藏中...({int(current_progress*100)}%)")
        
        # --- 關鍵修正：增加間隔時間，防止被 Rate Limit ---
        time.sleep(0.3)
    
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
            
            status_text = st.empty()
            progress_bar = st.progress(0, text="準備下載...")
            
            df = fetch_all_data(stock_dict, progress_bar, status_text)
            
            if not df.empty:
                df.to_csv(CACHE_FILE, index=False)
                st.success("💾 資料已儲存至快取！")
            
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
    min_vol_input = st.number_input("最低成交量 (張)", value=1000, step=100)
    
    st.subheader("策略選擇")
    
    strategy_mode = st.radio(
        "選擇篩選策略：",
        ("🛡️ 守護生命線 (反彈/支撐)", "🔥 浴火重生 (假跌破)", "👑 皇冠特選 (多頭排列)")
    )

    st.caption("細部條件：")
    
    filter_trend_up = False
    filter_trend_down = False
    filter_kd = False
    filter_vol_double = False
    filter_royal = False
    filter_treasure = False

    if strategy_mode == "🛡️ 守護生命線 (反彈/支撐)":
        col1, col2 = st.columns(2)
        with col1: filter_trend_up = st.checkbox("生命線向上", value=False)
        with col2: filter_trend_down = st.checkbox("生命線向下", value=False)
        filter_kd = st.checkbox("KD 黃金交叉", value=False)
        filter_vol_double = st.checkbox("出量 (今日 > 昨日x1.5)", value=False)
    
    elif strategy_mode == "🔥 浴火重生 (假跌破)":
        filter_treasure = True
        st.info("ℹ️ 尋找：過去7日內曾跌破，但今日站回生命線的個股。")
        filter_vol_double = st.checkbox("出量確認", value=False)

    elif strategy_mode == "👑 皇冠特選 (多頭排列)":
        filter_royal = True
        st.info("ℹ️ 條件：股價 > 20MA > 60MA > 200MA (多頭強勢股)")
        st.markdown("""
        **回測規則 (更嚴格)：**
        * **停利**：20天內任一天觸及 +10%
        * **停損**：收盤價跌破 200MA
        """)
        filter_vol_double = st.checkbox("出量確認", value=False)

    st.divider()
    
    st.caption("⚠️ 回測將使用上方設定的「最低成交量」進行過濾。")
    if st.button("🧪 策略回測"):
        st.info("阿吉正在調閱過去2年的歷史檔案，進行深度驗證... (請稍候) ⏳")
        stock_dict = get_stock_list()
        bt_progress = st.progress(0, text="初始化回測...")
        
        use_treasure_param = True if strategy_mode == "🔥 浴火重生 (假跌破)" else False
        use_royal_param = True if strategy_mode == "👑 皇冠特選 (多頭排列)" else False
        
        bt_df = run_strategy_backtest(
            stock_dict, 
            bt_progress, 
            use_trend_up=filter_trend_up, 
            use_treasure=use_treasure_param, 
            use_vol=filter_vol_double,
            use_royal=use_royal_param,
            min_vol_threshold=min_vol_input 
        )
        
        st.session_state['backtest_result'] = bt_df
        bt_progress.empty()
        st.success("回測完成！請查看下方結果。")

    with st.expander("📅 系統開發日誌"):
        st.markdown("""
        ### Ver 3.18 (Stability Fix)
        * **Fix**: 修正資料下載不全的問題 (Batch Size 150 -> 50，增加間隔時間)。
        * **Fix**: 修正「浴火重生」策略中，舊的歷史訊號會掩蓋昨日最新關注訊號的問題。
        * **Opt**: 維持本地快取機制 (CSV)。
        """)

# 主畫面 - 回測報告
if st.session_state['backtest_result'] is not None:
    bt_df = st.session_state['backtest_result']
    st.markdown("---")
    
    s_name = "🛡️ 守護生命線"
    if filter_treasure: s_name = "🔥 浴火重生"
    elif filter_royal: s_name = "👑 皇冠特選"
    
    st.subheader(f"🧪 策略回測報告：{s_name}")

    df_history = bt_df[bt_df['結果'] != "觀察中"].copy()
    df_watching = bt_df[bt_df['結果'] == "觀察中"].copy()
    
    # 1. 關注中
    if not df_watching.empty:
        st.markdown(f"""
        <div style="background-color: #fff8dc; padding: 15px; border-radius: 10px; border: 2px solid #ffa500; margin-bottom: 20px;">
            <h3 style="color: #d2691e; margin:0;">👀 旺來關注中 (進行中訊號)</h3>
            <p style="color: #666; margin:5px 0 0 0;">{'這些股票尚未觸發停利(+10%)或停損(破線)。' if filter_royal else '這些股票訊號發生未滿 10 天。'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        df_watching = df_watching.sort_values(by='訊號日期', ascending=False)
        st.dataframe(
            df_watching[['代號', '名稱', '訊號日期', '訊號價', '最高漲幅(%)']].style.background_gradient(cmap='Reds', subset=['最高漲幅(%)']),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("👀 目前沒有符合「關注中」的股票。")

    st.markdown("---")
    st.markdown("### 📜 歷史驗證數據 (已結算)")

    # 2. 歷史數據
    if len(df_history) > 0:
        months = sorted(df_history['月份'].unique())
        tabs = st.tabs(["📊 總覽"] + months)
        
        with tabs[0]:
            win_df = df_history[df_history['結果'].str.contains("Win") | df_history['結果'].str.contains("驗證成功")]
            win_count = len(win_df)
            total_count = len(df_history)
            win_rate = int((win_count / total_count) * 100) if total_count > 0 else 0
            avg_max_ret = round(df_history['最高漲幅(%)'].mean(), 2)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("總已結算次數", total_count)
            col2.metric("獲利機率", f"{win_rate}%")
            col3.metric("平均損益(%)", f"{avg_max_ret}%")
            st.dataframe(df_history, use_container_width=True)

        for i, m in enumerate(months):
            with tabs[i+1]:
                m_df = df_history[df_history['月份'] == m]
                m_win = len(m_df[m_df['結果'].str.contains("Win") | m_df['結果'].str.contains("驗證成功")])
                m_total = len(m_df)
                m_rate = int((m_win / m_total) * 100) if m_total > 0 else 0
                m_avg = round(m_df['最高漲幅(%)'].mean(), 2) if m_total > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{m} 結算次數", m_total)
                c2.metric(f"{m} 獲利機率", f"{m_rate}%")
                c3.metric(f"{m} 平均損益", f"{m_avg}%")
                
                def color_ret(val): return f'color: {"red" if val > 0 else "green"}'
                st.dataframe(m_df.style.map(color_ret, subset=['最高漲幅(%)']), use_container_width=True)
    else:
        st.warning("在此回測期間內，沒有歷史股票符合條件。")
    st.markdown("---")

# 主畫面 - 日常篩選
if st.session_state['master_df'] is not None:
    df = st.session_state['master_df'].copy()
    
    if '生命線' not in df.columns:
        st.error("⚠️ 資料結構已更新！請點擊 **「🚨 強制重置系統」** 後重新下載。")
        st.stop()

    # 基礎過濾
    df = df[df['成交量'] >= (min_vol_input * 1000)]
    
    # 策略分流篩選
    if strategy_mode == "🔥 浴火重生 (假跌破)":
        df = df[df['浴火重生'] == True]
    elif strategy_mode == "👑 皇冠特選 (多頭排列)":
        if '皇冠特選' in df.columns:
            df = df[df['皇冠特選'] == True]
        else:
            # 相容性處理
            df = df[(df['收盤價'] > df['MA20']) & (df['MA20'] > df['MA60']) & (df['MA60'] > df['生命線'])]
    else:
        # 守護生命線
        df = df[df['abs_bias'] <= bias_threshold]
        if filter_trend_up: df = df[df['生命線趨勢'] == "⬆️向上"]
        elif filter_trend_down: df = df[df['生命線趨勢'] == "⬇️向下"]
        if filter_kd: df = df[df['K值'] > df['D值']]
    
    if filter_vol_double: 
        df = df[df['成交量'] > (df['昨日成交量'] * 1.5)]
        
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
        df['選股標籤'] = df['代號'] + " " + df['名稱']
        
        display_cols = ['代號', '名稱', '收盤價', '生命線', '乖離率(%)', '位置', 'KD值', '成交量(張)']
        if strategy_mode == "👑 皇冠特選 (多頭排列)":
            display_cols = ['代號', '名稱', '收盤價', 'MA20', 'MA60', '生命線', 'KD值', '成交量(張)']
            
        df = df.sort_values(by='成交量', ascending=False)
        
        tab1, tab2 = st.tabs(["📋 篩選結果列表", "📊 日趨勢圖"])
        
        with tab1:
            def highlight_row(row):
                return ['background-color: #e6fffa; color: black'] * len(row) if row['收盤價'] > row['生命線'] else ['background-color: #fff0f0; color: black'] * len(row)

            st.dataframe(df[display_cols].style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("### 🔍 個股趨勢圖")
            if len(df) > 0:
                selected_stock_label = st.selectbox("請選擇一檔股票：", df['選股標籤'].tolist())
                selected_row = df[df['選股標籤'] == selected_stock_label].iloc[0]
                plot_stock_chart(selected_row['完整代號'], selected_row['名稱'])
                
                c1, c2, c3 = st.columns(3)
                # 這裡幫您把小數點修整為兩位
                c1.metric("收盤價", f"{selected_row['收盤價']:.2f}")
                c2.metric("成交量", f"{selected_row['成交量(張)']} 張")
                c3.metric("KD", selected_row['KD值'])

else:
    st.warning("👈 請先點擊左側 sidebar 的 **「🔄 下載最新股價」** 按鈕開始挖寶！")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("welcome.jpg"):
            st.markdown("""<div style="text-align: center; font-size: 1.1em; margin-bottom: 20px;">
                這是數年來的經驗收納<br>此工具僅供參考，不代表投資建議<br>預祝心想事成，從從容容，紫氣東來! 🟣✨</div>""", unsafe_allow_html=True)
            sub_c1, sub_c2, sub_c3 = st.columns([1, 1, 1])
            with sub_c2: st.image("welcome.jpg", width=180)
        else:
            st.info("💡 尚未偵測到 welcome.jpg")
```
