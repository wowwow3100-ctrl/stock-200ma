# --- 修正點：加入 show_spinner=False 以解決 Python 3.13 的執行緒錯誤 ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_list():
    """取得台股清單 (排除金融/ETF)"""
    try:
        # 強制更新 twstock 的股票代碼清單，避免抓到舊資料或空的
        twstock.__update_codes()
        
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
    except Exception as e:
        # 如果 twstock 連線失敗，回傳空字典，避免整個程式崩潰
        print(f"Error fetching stock list: {e}")
        return {}
