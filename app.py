# ==============================
# Streamlit Cloud Stable Version
# Stock AI Forecast Web App
# ==============================

import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------
# Streamlit 基本設定
# ------------------------------
st.set_page_config(
    page_title="股票 AI 預測",
    layout="wide"
)

st.title("📈 股票 AI 預測（雲端穩定版）")
st.caption("⚠️ 本系統為統計模型推估，非投資建議")

# 啟動保護（非常重要，避免雲端卡死）
st.info("✅ 系統已啟動，請輸入股票代號後再開始預測")
time.sleep(0.5)

# ------------------------------
# 使用者輸入
# ------------------------------
ticker = st.text_input(
    "請輸入股票代號（台股：2330 / 美股：AAPL）",
    placeholder="例如：2330 或 AAPL"
).strip()

market = st.radio(
    "市場別",
    ["台股", "美股"],
    horizontal=True
)

if market == "台股" and ticker.isdigit():
    ticker_symbol = f"{ticker}.TW"
else:
    ticker_symbol = ticker.upper()

# ------------------------------
# 預測參數
# ------------------------------
FORECAST_DAYS = {
    "T+1": 1,
    "T+5": 5,
    "T+10": 10,
    "1個月": 21,
    "半年": 126,
    "1年": 252
}

# ------------------------------
# 預測按鈕（所有重計算只在這裡跑）
# ------------------------------
if st.button("🚀 開始 AI 預測", type="primary"):

    try:
        with st.spinner("📥 下載股價資料中..."):
            df = yf.download(
                ticker_symbol,
                period="5y",
                interval="1d",
                progress=False
            )

        if df.empty:
            st.error("❌ 查無資料，請確認股票代號")
            st.stop()

        df = df.dropna()
        close = df["Close"]

        st.success(f"✅ 成功取得 {len(close)} 筆歷史資料")

        # --------------------------
        # SARIMAX 模型
        # --------------------------
        with st.spinner("🤖 AI 模型計算中（請稍候）..."):

            log_price = np.log(close)
            returns = log_price.diff().dropna()

            model = SARIMAX(
                returns,
                order=(1, 0, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            result = model.fit(disp=False)

        # --------------------------
        # 預測結果
        # --------------------------
        last_price = close.iloc[-1]
        sigma = np.std(result.resid)

        forecast_results = []

        for label, days in FORECAST_DAYS.items():
            forecast = result.get_forecast(steps=days)
            mean_return = forecast.predicted_mean.sum()

            predicted_price = last_price * np.exp(mean_return)

            forecast_results.append({
                "期間": label,
                "預測價格": round(float(predicted_price), 2)
            })

        result_df = pd.DataFrame(forecast_results)

        # --------------------------
        # 顯示結果
        # --------------------------
        st.subheader("📊 預測結果表")
        st.dataframe(result_df, use_container_width=True)

        # 歷史走勢圖
        st.subheader("📈 歷史股價走勢")
        st.line_chart(close)

        # --------------------------
        # 趨勢文字說明
        # --------------------------
        trend_text = []

        if result_df["預測價格"].iloc[-1] > last_price:
            trend_text.append("🔼 模型顯示中長期趨勢偏多")
        else:
            trend_text.append("🔽 模型顯示中長期趨勢偏保守")

        volatility = np.std(returns) * np.sqrt(252)

        if volatility > 0.4:
            trend_text.append("⚠️ 波動度偏高，風險較大")
        else:
            trend_text.append("📉 波動度屬中低區間")

        st.subheader("🧠 趨勢解讀（AI 產生）")
        for t in trend_text:
            st.write("-", t)

        st.success("🎉 預測完成")

    except Exception as e:
        st.error("❌ 發生錯誤")
        st.exception(e)

else:
    st.info("⬆️ 請先輸入股票代號，然後點擊「開始 AI 預測」")
