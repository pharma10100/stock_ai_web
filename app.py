# app.py
import re
import math
import time
import json
import textwrap
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz

import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# Page config + simple "fullscreen-like" clean UI
# -----------------------------
st.set_page_config(
    page_title="股票AI預測｜未來走勢推估",
    layout="wide",
    page_icon="📈",
)

st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
      .block-container {padding-top: 1rem; padding-bottom: 2rem;}
      @media (max-width: 640px) {
        .block-container {padding-left: 0.9rem; padding-right: 0.9rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utilities
# -----------------------------
TZ_TAIPEI = timezone(timedelta(hours=8))

def is_taiwan_code(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4,6}", s.strip()))

def try_yf_download(ticker: str, period="10y", interval="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(ttl=24*3600)
def fetch_tw_stock_list() -> pd.DataFrame:
    """
    Fetch TWSE + TPEx lists (best-effort).
    - TWSE: https://openapi.twse.com.tw/v1/opendata/t187ap03_L  (often used)
    - TPEx:  public endpoints can change; we fall back gracefully.
    """
    rows = []
    # TWSE open data (commonly used)
    try:
        url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        for it in data:
            code = str(it.get("公司代號", "")).strip()
            name = str(it.get("公司簡稱", "")).strip()
            market = "TWSE"
            if code and name:
                rows.append({"code": code, "name": name, "market": market, "yf": f"{code}.TW"})
    except Exception:
        pass

    # TPEx (try a couple known patterns; if fails, still usable for TWSE)
    # You can replace this with your preferred stable TPEx source if you have one.
    tpex_candidates = [
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
        "https://www.tpex.org.tw/openapi/v1/t187ap03_O",
    ]
    for url in tpex_candidates:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json()
            for it in data:
                code = str(it.get("公司代號", "")).strip()
                name = str(it.get("公司簡稱", "")).strip()
                if code and name:
                    rows.append({"code": code, "name": name, "market": "TPEx", "yf": f"{code}.TWO"})
            break
        except Exception:
            continue

    df = pd.DataFrame(rows).drop_duplicates(subset=["code", "market"])
    return df

def fuzzy_pick_symbol(query: str, tw_list: pd.DataFrame):
    q = query.strip()
    # If user typed numeric code -> direct TW
    if is_taiwan_code(q):
        code = q
        # prefer TWSE .TW; if empty data then try .TWO later
        return {
            "display": f"{code}（台股）",
            "yf_candidates": [f"{code}.TW", f"{code}.TWO"],
            "meta": {"market_guess": "TW", "code": code}
        }

    # If looks like ticker (letters/dots) -> treat as US/global via yfinance
    if re.fullmatch(r"[A-Za-z\.\-\^]{1,12}", q):
        return {
            "display": f"{q.upper()}（Ticker）",
            "yf_candidates": [q.upper()],
            "meta": {"market_guess": "US", "ticker": q.upper()}
        }

    # Otherwise try TW list fuzzy match by name or code string
    if tw_list is not None and not tw_list.empty:
        choices = (tw_list["code"] + " " + tw_list["name"] + " " + tw_list["market"]).tolist()
        best = process.extract(q, choices, scorer=fuzz.WRatio, limit=8)
        return best  # list of tuples (choice, score, idx)
    return []

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d["Volume"] = pd.to_numeric(d.get("Volume", np.nan), errors="coerce")

    # Moving averages
    for w in [5, 10, 20, 60, 120, 240]:
        d[f"MA{w}"] = d["Close"].rolling(w).mean()

    # RSI(14)
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI14"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    # Bollinger Bands (20, 2)
    mid = d["Close"].rolling(20).mean()
    sd = d["Close"].rolling(20).std()
    d["BB_mid"] = mid
    d["BB_up"] = mid + 2 * sd
    d["BB_low"] = mid - 2 * sd

    # Volume trend
    d["VolMA20"] = d["Volume"].rolling(20).mean()
    return d

def fit_sarimax_forecast(close: pd.Series, steps: int):
    """
    SARIMAX on log returns; output distribution by:
    - model mean forecast on returns
    - volatility estimated from residuals
    - Monte Carlo to create price path distribution
    """
    close = close.dropna().astype(float)
    if len(close) < 200:
        raise ValueError("歷史資料不足（建議至少 200 個交易日）")

    logp = np.log(close)
    ret = logp.diff().dropna()

    # Simple but robust SARIMAX; keep it lightweight
    model = SARIMAX(ret, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # Forecast mean returns
    fc = res.get_forecast(steps=steps)
    mean_ret = fc.predicted_mean.values

    # Residual volatility
    resid = res.resid
    sigma = float(np.nanstd(resid)) if np.isfinite(np.nanstd(resid)) else float(np.nanstd(ret))
    sigma = max(sigma, 1e-4)

    last_price = float(close.iloc[-1])

    # Monte Carlo simulate
    n_sims = 4000 if steps <= 260 else 2000
    rng = np.random.default_rng(42)
    shocks = rng.normal(0, sigma, size=(n_sims, steps))
    sim_rets = mean_ret.reshape(1, -1) + shocks
    sim_logp = np.log(last_price) + np.cumsum(sim_rets, axis=1)
    sim_prices = np.exp(sim_logp)

    # median + 10~90% interval
    med = np.median(sim_prices, axis=0)
    p10 = np.percentile(sim_prices, 10, axis=0)
    p90 = np.percentile(sim_prices, 90, axis=0)
    return med, p10, p90

def horizons_to_steps():
    # Use trading-day approximations
    # T+1..T+10 = 1..10
    # 1週=5, 1月=21, 半年=126, 1年=252, 3年=756, 10年=2520
    h = {f"T+{i}": i for i in range(1, 11)}
    h.update({
        "1週內": 5,
        "1個月內": 21,
        "半年": 126,
        "1年": 252,
        "3年": 756,
        "10年": 2520
    })
    return h

def google_news_rss(query: str, days: int = 30, lang="zh-TW", region="TW"):
    """
    Google News RSS doesn't require API key.
    We'll fetch and then filter by publish date (best-effort).
    """
    q = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={q}%20when:{days}d&hl={lang}&gl={region}&ceid={region}:{lang}"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:25]:
        title = getattr(e, "title", "")
        link = getattr(e, "link", "")
        published = getattr(e, "published", "")
        summary = getattr(e, "summary", "")

        # Clean summary html
        try:
            summary_txt = BeautifulSoup(summary, "lxml").get_text(" ", strip=True)
        except Exception:
            summary_txt = re.sub(r"<.*?>", "", summary)

        items.append({
            "title": title,
            "link": link,
            "published": published,
            "summary": summary_txt
        })
    return items

def simple_news_sentiment(items: list) -> float:
    """
    Ultra-light sentiment scoring using keyword heuristics (Chinese + finance).
    Return score in [-1, 1].
    """
    pos_kw = ["利多", "上修", "成長", "創高", "獲利", "強勁", "擴產", "合作", "訂單", "買回庫藏股", "調升目標價"]
    neg_kw = ["利空", "下修", "衰退", "創低", "虧損", "疲弱", "裁員", "減產", "延遲", "下調目標價", "罰款", "調查", "風險"]
    score = 0.0
    text = " ".join([(it.get("title","") + " " + it.get("summary","")) for it in items]).lower()

    for w in pos_kw:
        score += 0.08 * text.count(w)
    for w in neg_kw:
        score -= 0.10 * text.count(w)
    return float(np.tanh(score))

def build_reason_text(df_ind: pd.DataFrame, news_items: list, news_score: float, info: dict):
    last = df_ind.dropna().iloc[-1]
    close = float(last["Close"])
    ma20 = float(last.get("MA20", np.nan))
    ma60 = float(last.get("MA60", np.nan))
    rsi = float(last.get("RSI14", np.nan))
    macd = float(last.get("MACD", np.nan))
    macd_hist = float(last.get("MACD_hist", np.nan))
    vol = float(last.get("Volume", np.nan))
    volma = float(last.get("VolMA20", np.nan))

    parts = []

    # Technical
    tech = []
    if np.isfinite(ma20) and np.isfinite(ma60):
        if close > ma20 > ma60:
            tech.append("股價位於 MA20、MA60 之上，偏多排列")
        elif close < ma20 < ma60:
            tech.append("股價位於 MA20、MA60 之下，偏空排列")
        else:
            tech.append("均線呈現糾結/過渡型態，短線方向仍需確認")

    if np.isfinite(rsi):
        if rsi >= 70:
            tech.append(f"RSI(14)≈{rsi:.1f} 偏高，短線可能過熱、易震盪")
        elif rsi <= 30:
            tech.append(f"RSI(14)≈{rsi:.1f} 偏低，可能接近超賣區")
        else:
            tech.append(f"RSI(14)≈{rsi:.1f} 屬中性區間")

    if np.isfinite(macd) and np.isfinite(macd_hist):
        if macd_hist > 0:
            tech.append("MACD 柱狀體為正，動能偏多")
        else:
            tech.append("MACD 柱狀體為負，動能偏弱")

    # Volume
    vol_txt = None
    if np.isfinite(vol) and np.isfinite(volma) and volma > 0:
        ratio = vol / volma
        if ratio >= 1.5:
            vol_txt = "量能明顯放大（>20日均量），趨勢延續機率提升但也可能伴隨劇烈波動"
        elif ratio <= 0.7:
            vol_txt = "量能偏低（<20日均量），較可能進入盤整/等待催化劑"
        else:
            vol_txt = "量能接近常態（約等於20日均量）"

    # News
    if news_items:
        if news_score > 0.25:
            news_txt = "近 1 個月新聞語氣偏正向（利多/成長/訂單等詞彙較多）"
        elif news_score < -0.25:
            news_txt = "近 1 個月新聞語氣偏負向（利空/下修/風險等詞彙較多）"
        else:
            news_txt = "近 1 個月新聞語氣中性偏混合（利多利空交錯）"
    else:
        news_txt = "近 1 個月新聞抓取較少，新聞面權重降低（可能是關鍵字太短或公司曝光度較低）"

    # Fundamentals (best-effort)
    fund = []
    if info:
        pe = info.get("trailingPE") or info.get("forwardPE")
        mcap = info.get("marketCap")
        eps = info.get("trailingEps")
        revg = info.get("revenueGrowth")

        if pe is not None:
            fund.append(f"本益比(PE)約 {pe:.1f}（yfinance）")
        if eps is not None:
            fund.append(f"EPS 約 {eps:.2f}（yfinance）")
        if revg is not None and isinstance(revg, (int, float)) and np.isfinite(revg):
            fund.append(f"營收成長率約 {revg*100:.1f}%（yfinance）")
        if mcap is not None:
            # human readable
            if mcap >= 1e12:
                fund.append(f"市值約 {mcap/1e12:.2f} 兆")
            elif mcap >= 1e9:
                fund.append(f"市值約 {mcap/1e9:.2f} 十億")
            else:
                fund.append(f"市值約 {mcap:.0f}")

    parts.append("【技術面】" + "；".join(tech) if tech else "【技術面】資料不足")
    if vol_txt:
        parts.append("【量能】" + vol_txt)
    parts.append("【新聞面】" + news_txt)
    if fund:
        parts.append("【基本面(可得資料)】" + "；".join(fund))

    parts.append("【提醒】預測為統計模型推估，長天期（半年以上）誤差會快速放大，請務必搭配風險控管。")
    return "\n".join(parts)

# -----------------------------
# UI
# -----------------------------
st.title("📈 股票AI預測｜未來走勢推估（免安裝・分享網址即可用）")

with st.sidebar:
    st.header("設定")
    period = st.selectbox("歷史資料範圍（用於模型訓練）", ["1y", "3y", "5y", "10y", "max"], index=3)
    st.caption("建議至少 3y～10y，短期資料太少會影響穩定度。")

    show_interval = st.selectbox("歷史K線粒度", ["1d"], index=0, disabled=True)
    st.write("---")
    st.caption("全螢幕：瀏覽器按 **F11**（手機可用瀏覽器選單『全螢幕』或橫向顯示）。")

query = st.text_input("輸入股票代號或股票名稱（支援模糊搜尋）", placeholder="例：2330、台積電、AAPL、TSLA")

tw_list = fetch_tw_stock_list()

selected_yf = None
display_name = None

if query.strip():
    pick = fuzzy_pick_symbol(query, tw_list)

    # Case A: direct ticker or TW code
    if isinstance(pick, dict):
        yf_candidates = pick["yf_candidates"]
        display_name = pick["display"]

        # Try download to confirm
        ok = None
        df_hist = pd.DataFrame()
        used = None
        for cand in yf_candidates:
            df_hist = try_yf_download(cand, period=period, interval="1d")
            if not df_hist.empty:
                ok = True
                used = cand
                break
        if ok:
            selected_yf = used
        else:
            st.error("找不到可用的股價資料：請確認代號/市場是否正確。台股請用 4 碼代號，美股請用 ticker。")
    else:
        # Case B: fuzzy list results
        st.subheader("猜你想找的是（點選一個）")
        if len(pick) == 0:
            st.warning("沒有匹配到台股清單。你也可以直接輸入美股 ticker（如 AAPL）。")
        else:
            opts = []
            for choice, score, idx in pick:
                row = tw_list.iloc[idx]
                opts.append((f"{row['code']} {row['name']}（{row['market']}）｜相似度 {score}", row["yf"]))
            chosen = st.radio("選擇股票", options=[o[0] for o in opts], index=0)
            selected_yf = dict(opts)[chosen]
            display_name = chosen.split("｜")[0].strip()

if selected_yf:
    st.success(f"已選擇：**{display_name}** → yfinance：`{selected_yf}`")

    # Load data
    df = try_yf_download(selected_yf, period=period, interval="1d")
    if df.empty:
        st.error("股價資料抓取失敗（可能是代號、網路或資料源限制）。請換一個代號再試。")
        st.stop()

    df = df.dropna(subset=["Close"])
    df_ind = compute_indicators(df)

    # Get info (best-effort)
    info = {}
    try:
        info = yf.Ticker(selected_yf).info or {}
    except Exception:
        info = {}

    # News (last 30 days)
    news_q = display_name.split("（")[0].strip()
    news_items = []
    try:
        news_items = google_news_rss(news_q, days=30, lang="zh-TW", region="TW")
    except Exception:
        news_items = []

    news_score = simple_news_sentiment(news_items) if news_items else 0.0

    # Forecast horizons
    hs = horizons_to_steps()
    max_steps = max(hs.values())

    # Fit + forecast
    close_series = df_ind["Close"]
    try:
        med, p10, p90 = fit_sarimax_forecast(close_series, steps=max_steps)
    except Exception as e:
        st.error(f"模型預測失敗：{e}")
        st.stop()

    last_date = pd.to_datetime(df_ind["Date"].iloc[-1])
    last_close = float(df_ind["Close"].iloc[-1])

    # Build forecast table (16 cols)
    rows = []
    for label, step in hs.items():
        rows.append({
            "期間": label,
            "預測價(中位數)": float(med[step-1]),
            "可能區間(10%~90%)": f"{p10[step-1]:.2f} ~ {p90[step-1]:.2f}",
            "距離(交易日)": step
        })
    out = pd.DataFrame(rows)

    # Layout
    colA, colB = st.columns([1.2, 0.8], gap="large")

    with colA:
        st.subheader("走勢圖（歷史 + 預測）")

        # Build plot df
        hist_plot = df_ind[["Date", "Close"]].copy()
        # Forecast index as "steps" after last date; for visualization use incremental integers
        f_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=max_steps, freq="B")
        fc_plot = pd.DataFrame({
            "Date": f_dates,
            "Forecast_Median": med,
            "Forecast_P10": p10,
            "Forecast_P90": p90
        })

        plot_df = hist_plot.tail(700)  # keep it responsive
        st.line_chart(
            data=pd.DataFrame({"Close": plot_df["Close"].values}, index=plot_df["Date"]),
            height=380
        )

        st.caption("下方為預測區間（中位數 + 10%~90% 可能範圍）")
        # Show forecast chart
        fc_show = fc_plot.copy().set_index("Date")
        st.line_chart(fc_show[["Forecast_Median"]], height=260)
        st.area_chart(fc_show[["Forecast_P10", "Forecast_P90"]], height=180)

    with colB:
        st.subheader("16 欄位：未來可能股價")
        st.dataframe(
            out[["期間", "預測價(中位數)", "可能區間(10%~90%)"]],
            use_container_width=True,
            height=520
        )

        st.markdown("**最新收盤價**：" + f"{last_close:.2f}")
        st.markdown("**新聞情緒分數（近 30 天）**：" + f"{news_score:+.2f}（-1 負向 / +1 正向）")

    # News list
    st.write("---")
    st.subheader("近 1 個月新聞（摘要）")
    if news_items:
        for it in news_items[:10]:
            st.markdown(f"- **{it['title']}**  \n  {it['summary'][:160]}…")
    else:
        st.info("目前抓不到新聞（可能關鍵字太短、或資料源限制）。你可以改用：『公司全名 + 代號』再試。")

    # Reason text
    st.write("---")
    st.subheader("趨勢原因（自動生成）")
    reason = build_reason_text(df_ind, news_items, news_score, info)
    st.text(reason)

    # Download results
    st.write("---")
    st.subheader("匯出")
    csv = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載預測表（CSV）", data=csv, file_name="forecast_table.csv", mime="text/csv")

else:
    st.info("請先在上方輸入股票代號或名稱。")
