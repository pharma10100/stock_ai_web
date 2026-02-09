# app.py — Streamlit Cloud "startup-safe" version
# 核心策略：啟動時只 import streamlit；所有重套件/網路抓取/模型運算都延遲到按鈕才做。

import streamlit as st
import time

st.set_page_config(page_title="股票 AI 預測", layout="wide")
st.title("📈 股票 AI 預測（雲端啟動保險版）")
st.caption("⚠️ 本工具為統計推估，不構成投資建議。長天期（3年/10年）不確定性很高。")

# 讓 Cloud 先成功 render（避免啟動階段卡死）
st.info("✅ 系統已啟動：請輸入股票代號/名稱後，按「開始預測」才會抓資料與計算。")
time.sleep(0.2)

# ----------------------------
# 輕量 UI（不做任何重計算）
# ----------------------------
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    q = st.text_input(
        "輸入股票代號或名稱（先以代號為主；台股輸入 2330、美股輸入 AAPL）",
        placeholder="例：2330、2317、AAPL、TSLA"
    ).strip()

with col2:
    market = st.radio("市場", ["台股", "美股/其他"], horizontal=True)
    period = st.selectbox("訓練用歷史資料範圍", ["1y", "3y", "5y", "10y", "max"], index=3)
    st.caption("提示：台股代號會優先用 .TW，抓不到再試 .TWO。")

# 16 欄位需求
HORIZONS = {f"T+{i}": i for i in range(1, 11)}
HORIZONS.update({"1週內": 5, "1個月內": 21, "半年": 126, "1年": 252, "3年": 756, "10年": 2520})

def _format_ticker(query: str, market_choice: str):
    query = (query or "").strip()
    if not query:
        return None, None
    if market_choice == "台股" and query.isdigit():
        # 先 .TW，失敗再 .TWO
        return query, [f"{query}.TW", f"{query}.TWO"]
    # 其他：直接當作 ticker
    return query.upper(), [query.upper()]

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# ----------------------------
# 主按鈕：重套件與運算全部在這裡
# ----------------------------
if st.button("🚀 開始預測（抓資料 + 建模）", type="primary", use_container_width=True):
    if not q:
        st.error("請先輸入股票代號或名稱。")
        st.stop()

    display, candidates = _format_ticker(q, market)
    st.write(f"✅ 目標：**{display}**")

    # 延遲 import（重點）
    with st.spinner("載入套件中…（第一次雲端啟動可能稍久）"):
        import numpy as np
        import pandas as pd

    # 下載股價（延遲 import）
    with st.spinner("下載歷史股價中…"):
        import yfinance as yf

        df = None
        used = None
        for c in candidates:
            try:
                tmp = yf.download(c, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
                if tmp is not None and not tmp.empty:
                    df = tmp.copy()
                    used = c
                    break
            except Exception:
                continue

        if df is None or df.empty:
            st.error("抓不到股價資料：請確認代號/市場是否正確（台股請用數字代號，美股用 ticker）。")
            st.stop()

    st.success(f"已抓到資料：`{used}`，共 {len(df)} 筆日資料")

    # 整理資料
    df = df.dropna()
    df = df.reset_index()
    if "Date" not in df.columns:
        # yfinance 新版可能是 Datetime
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df.get("Volume", np.nan), errors="coerce")
    df = df.dropna(subset=["Close"])
    if len(df) < 220:
        st.warning("歷史資料偏少（< 220 交易日），預測穩定性可能較差。")

    # 指標（技術面 + 量能）
    d = df.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA60"] = d["Close"].rolling(60).mean()
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI14"] = 100 - (100 / (1 + rs))
    d["VolMA20"] = d["Volume"].rolling(20).mean()

    last_close = float(d["Close"].iloc[-1])

    # 取最大步數預測
    max_steps = max(HORIZONS.values())

    # 建模：優先 SARIMAX；若失敗改用「漂移+波動」簡易模型（保底不當機）
    with st.spinner("建模與預測中…"):
        med = p10 = p90 = None
        model_used = None

        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            close = d["Close"].astype(float)
            logp = np.log(close)
            ret = logp.diff().dropna()

            if len(ret) < 200:
                raise ValueError("報酬序列太短")

            m = SARIMAX(
                ret,
                order=(1, 0, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = m.fit(disp=False)

            fc = res.get_forecast(steps=max_steps)
            mean_ret = fc.predicted_mean.values

            sigma = float(np.nanstd(res.resid))
            sigma = max(sigma, 1e-4)

            last_price = float(close.iloc[-1])

            n_sims = 3000 if max_steps <= 756 else 1500
            rng = np.random.default_rng(42)
            shocks = rng.normal(0, sigma, size=(n_sims, max_steps))
            sim_rets = mean_ret.reshape(1, -1) + shocks
            sim_logp = np.log(last_price) + np.cumsum(sim_rets, axis=1)
            sim_prices = np.exp(sim_logp)

            med = np.median(sim_prices, axis=0)
            p10 = np.percentile(sim_prices, 10, axis=0)
            p90 = np.percentile(sim_prices, 90, axis=0)
            model_used = "SARIMAX(1,0,1)+MonteCarlo"

        except Exception:
            # 保底模型：幾何布朗運動（漂移取近 60 日平均報酬，波動取近 60 日標準差）
            close = d["Close"].astype(float).values
            logp = np.log(close)
            ret = np.diff(logp)
            window = min(60, len(ret))
            mu = float(np.nanmean(ret[-window:])) if window > 5 else 0.0
            sigma = float(np.nanstd(ret[-window:])) if window > 5 else float(np.nanstd(ret))
            sigma = max(sigma, 1e-4)

            last_price = float(close[-1])
            n_sims = 3000 if max_steps <= 756 else 1500
            rng = np.random.default_rng(42)
            shocks = rng.normal(0, sigma, size=(n_sims, max_steps))
            sim_rets = mu + shocks
            sim_logp = np.log(last_price) + np.cumsum(sim_rets, axis=1)
            sim_prices = np.exp(sim_logp)

            med = np.median(sim_prices, axis=0)
            p10 = np.percentile(sim_prices, 10, axis=0)
            p90 = np.percentile(sim_prices, 90, axis=0)
            model_used = "GBM(漂移+波動)保底模型"

    st.success(f"預測完成（模型：{model_used}）")

    # 16 欄位表格
    rows = []
    for label, step in HORIZONS.items():
        rows.append({
            "期間": label,
            "預測價(中位數)": round(float(med[step-1]), 2),
            "可能區間(10~90%)": f"{p10[step-1]:.2f} ~ {p90[step-1]:.2f}",
        })
    out = pd.DataFrame(rows)

    # 圖表
    cA, cB = st.columns([1.4, 0.6], gap="large")

    with cA:
        st.subheader("📈 歷史股價走勢")
        hist = d[["Date", "Close"]].copy().set_index("Date")
        st.line_chart(hist["Close"], height=320)

        st.subheader("🔮 預測走勢（中位數 + 區間）")
        # 用工作日序列當作未來索引（只是視覺化）
        last_dt = pd.to_datetime(d["Date"].iloc[-1])
        future_idx = pd.date_range(last_dt + pd.Timedelta(days=1), periods=max_steps, freq="B")
        fc_df = pd.DataFrame({"Median": med, "P10": p10, "P90": p90}, index=future_idx)
        st.line_chart(fc_df[["Median"]], height=240)
        st.area_chart(fc_df[["P10", "P90"]], height=180)

    with cB:
        st.subheader("🧾 16 欄位預測表")
        st.dataframe(out, use_container_width=True, height=520)
        st.markdown(f"**最新收盤價**：{last_close:.2f}")

    # 趨勢原因文字（技術 + 量能 +（可選）新聞）
    st.write("---")
    st.subheader("🧠 趨勢原因（自動生成）")

    last = d.dropna().iloc[-1] if len(d.dropna()) else d.iloc[-1]
    close_now = float(last["Close"])
    ma20 = _safe_float(last.get("MA20"))
    ma60 = _safe_float(last.get("MA60"))
    rsi = _safe_float(last.get("RSI14"))
    vol = _safe_float(last.get("Volume"))
    volma = _safe_float(last.get("VolMA20"))

    reasons = []

    # 技術面
    tech_bits = []
    if ma20 and ma60:
        if close_now > ma20 > ma60:
            tech_bits.append("股價位於 MA20、MA60 之上（偏多排列）")
        elif close_now < ma20 < ma60:
            tech_bits.append("股價位於 MA20、MA60 之下（偏空排列）")
        else:
            tech_bits.append("均線糾結（可能盤整或轉折中）")
    if rsi is not None:
        if rsi >= 70:
            tech_bits.append(f"RSI≈{rsi:.1f} 偏高（短線過熱、易震盪）")
        elif rsi <= 30:
            tech_bits.append(f"RSI≈{rsi:.1f} 偏低（接近超賣）")
        else:
            tech_bits.append(f"RSI≈{rsi:.1f} 中性")

    if tech_bits:
        reasons.append("【技術面】" + "；".join(tech_bits))

    # 量能
    if vol is not None and volma and volma > 0:
        ratio = vol / volma
        if ratio >= 1.5:
            reasons.append("【量能】量能明顯放大（>20日均量），趨勢延續機率提升但波動可能加劇")
        elif ratio <= 0.7:
            reasons.append("【量能】量能偏低（<20日均量），較可能盤整、等待催化事件")
        else:
            reasons.append("【量能】量能接近常態（約等於20日均量）")

    # 新聞面（可選，若 requests/feedparser 不可用就跳過，不影響主流程）
    news_note = None
    try:
        import requests
        import feedparser
        from bs4 import BeautifulSoup  # 不用 lxml，使用內建 parser

        q_news = display
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(q_news)}%20when:30d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        r = requests.get(url, timeout=12)
        feed = feedparser.parse(r.text)

        titles = []
        for e in feed.entries[:8]:
            t = getattr(e, "title", "")
            s = getattr(e, "summary", "")
            # 用內建 html parser 清理
            try:
                s_txt = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
            except Exception:
                s_txt = ""
            titles.append((t + " " + s_txt).lower())

        text = " ".join(titles)
        pos_kw = ["利多", "上修", "成長", "創高", "獲利", "擴產", "合作", "訂單", "調升目標價"]
        neg_kw = ["利空", "下修", "衰退", "創低", "虧損", "減產", "裁員", "延遲", "下調目標價", "風險"]

        score = 0.0
        for w in pos_kw:
            score += 0.08 * text.count(w)
        for w in neg_kw:
            score -= 0.10 * text.count(w)
        score = float(np.tanh(score))

        if score > 0.25:
            news_note = "【新聞面】近 30 天新聞語氣偏正向（利多/成長/訂單類訊號較多）"
        elif score < -0.25:
            news_note = "【新聞面】近 30 天新聞語氣偏負向（利空/下修/風險類訊號較多）"
        else:
            news_note = "【新聞面】近 30 天新聞語氣中性偏混合（利多利空交錯）"

        st.caption("近 30 天新聞摘要（前 8 則）")
        for e in feed.entries[:8]:
            st.markdown(f"- {getattr(e, 'title', '')}")

    except Exception:
        news_note = "【新聞面】目前無法穩定抓取新聞（不影響預測主流程）"

    if news_note:
        reasons.append(news_note)

    reasons.append("【提醒】預測為統計推估，長天期誤差會放大，請搭配風險控管。")

    for line in reasons:
        st.write(line)

    # 匯出
    st.write("---")
    csv = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載預測表（CSV）", data=csv, file_name="forecast_table.csv", mime="text/csv")

else:
    st.info("⬆️ 輸入代號後按「開始預測」，才會進行抓資料與計算。")
