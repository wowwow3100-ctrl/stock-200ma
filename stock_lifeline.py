import streamlit as st
import yfinance as yf
import pandas as pd

# 設定頁面標題
st.title("200MA 策略回測工具")
st.write("策略：接近 200MA (3%內) 進場 -> 檢測「短線3%反彈」與「站穩5天後波段」")

# 1. 使用者輸入參數
col1, col2 = st.columns(2)
with col1:
    stock_id = st.text_input("輸入股票代碼 (如 2330.TW)", "2330.TW")
with col2:
    start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))

# 按鈕觸發
if st.button("執行回測"):
    st.info(f"正在下載 {stock_id} 資料並進行運算...")
    
    try:
        # 2. 取得資料
        df = yf.download(stock_id, start=start_date, progress=False)

        # 【關鍵修復】：處理 yfinance 新版的多層索引 (MultiIndex) 問題
        # 這一步能解決 "ValueError: Can only compare identically-labeled Series objects"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 檢查資料是否為空
        if df.empty:
            st.error("下載不到資料，請檢查股票代碼是否正確 (台股需加 .TW)")
            st.stop()

        # 計算 200MA
        df['200MA'] = df['Close'].rolling(window=200).mean()
        df = df.dropna() # 移除沒有 MA 的前 200 天

        # 策略參數
        entry_range_pct = 0.03   # 接近範圍 3%
        profit_target_pct = 0.03 # 短線獲利目標 3%
        stable_days = 5          # 站穩觀察天數 (一週)
        trend_period = 20        # 趨勢觀察天數 (一個月)

        results = []

        # 3. 逐日回測邏輯
        # 預留足夠的天數供後續驗證 (-25天)
        for i in range(len(df) - trend_period - stable_days):
            today = df.iloc[i]
            current_date = df.index[i]
            
            # 強制轉型為 float，避免 Series 比較錯誤
            try:
                close_price = float(today['Close'])
                ma_price = float(today['200MA'])
            except Exception as e:
                continue # 跳過資料異常的天數

            # --- 進場條件：股價在 200MA 上方且距離 < 3% ---
            if ma_price < close_price <= ma_price * (1 + entry_range_pct):
                
                # 取得未來區段資料
                future_5_days = df.iloc[i+1 : i+1+stable_days]
                future_month = df.iloc[i+1+stable_days : i+1+stable_days+trend_period]
                
                # --- 判斷 A：短線反彈獲利 (一周內最高價是否漲 3%) ---
                target_price = close_price * (1 + profit_target_pct)
                max_high_in_5_days = float(future_5_days['High'].max())
                is_short_term_win = max_high_in_5_days >= target_price
                
                # --- 判斷 B：是否站穩 (一周內收盤價都沒跌破當日 200MA) ---
                is_stable = True
                for j in range(len(future_5_days)):
                    d_row = future_5_days.iloc[j]
                    d_close = float(d_row['Close'])
                    d_ma = float(d_row['200MA'])
                    
                    if d_close < d_ma: # 只要有一天跌破
                        is_stable = False
                        break
                
                # 計算站穩後的績效 (僅當站穩時計算)
                trend_return = None
                if is_stable and not future_month.empty:
                    price_at_stable_end = float(future_5_days.iloc[-1]['Close'])
                    price_after_month = float(future_month.iloc[-1]['Close'])
                    trend_return = (price_after_month - price_at_stable_end) / price_at_stable_end

                results.append({
                    "日期": current_date.strftime('%Y-%m-%d'),
                    "收盤價": round(close_price, 2),
                    "200MA": round(ma_price, 2),
                    "短線成功(3%)": "是" if is_short_term_win else "否",
                    "是否站穩5天": "是" if is_stable else "否",
                    "站穩後月漲跌幅": trend_return # 保留數值以便計算
                })

        # 4. 統計與顯示結果
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            
            # --- 數據統計 ---
            total_trades = len(results_df)
            short_term_wins = results_df[results_df["短線成功(3%)"] == "是"].shape[0]
            short_term_win_rate = (short_term_wins / total_trades) * 100
            
            stable_df = results_df[results_df["是否站穩5天"] == "是"].copy()
            stable_count = len(stable_df)
            
            # 顯示主要指標
            st.success(f"回測完成！共觸發進場 {total_trades} 次")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("1. 短線(3%)勝率", f"{short_term_win_rate:.1f}%")
            with col_res2:
                st.metric("2. 成功站穩5天次數", f"{stable_count} 次")
            
            # 計算站穩後的趨勢表現
            if stable_count > 0:
                # 清洗掉資料不足導致為 None 的行
                valid_trends = stable_df.dropna(subset=['站穩後月漲跌幅'])
                if not valid_trends.empty:
                    avg_return = valid_trends['站穩後月漲跌幅'].mean() * 100
                    pos_trends = valid_trends[valid_trends['站穩後月漲跌幅'] > 0]
                    trend_win_rate = (len(pos_trends) / len(valid_trends)) * 100
                    
                    with col_res3:
                        st.metric("3. 站穩後-下月上漲機率", f"{trend_win_rate:.1f}%")
                    
                    st.write(f"**站穩後，持有 20 天的平均報酬率：** {avg_return:.2f}%")
                else:
                    with col_res3:
                        st.metric("3. 站穩後表現", "資料不足")
            else:
                with col_res3:
                    st.metric("3. 站穩後表現", "無站穩案例")

            st.divider()
            
            # 顯示詳細交易紀錄 (將數值轉為百分比顯示)
            st.subheader("詳細交易紀錄")
            
            # 格式化顯示用
            display_df = results_df.copy()
            display_df['站穩後月漲跌幅'] = display_df['站穩後月漲跌幅'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-"
            )
            
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.warning("在此期間內，沒有任何一天符合「接近 200MA 3%以內」的進場條件。")

    except Exception as e:
        st.error(f"發生錯誤: {e}")
        # 顯示詳細錯誤以便除錯
        import traceback
        st.text(traceback.format_exc())
