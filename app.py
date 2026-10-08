
import io
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# ============================================================
# GPW DZIDA — Advanced Scanner
# ============================================================

st.set_page_config(
    page_title="GPW DZIDA — Advanced Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_SYMBOLS = """PKO.WA PEO.WA PZU.WA PKN.WA KGH.WA SPL.WA ALE.WA DNP.WA CDR.WA XTB.WA LPP.WA CCC.WA JSW.WA KRU.WA ACP.WA BDX.WA ALR.WA BHW.WA ENA.WA ENG.WA PGE.WA TPE.WA CPS.WA DVL.WA MIL.WA MBK.WA KER.WA CAR.WA WPL.WA 11B.WA TEN.WA HUG.WA VRG.WA AMC.WA EAT.WA TXT.WA STX.WA RBW.WA ASB.WA ANR.WA ABE.WA MRC.WA ATT.WA GPP.WA GPW.WA MIR.WA NEU.WA SLV.WA 1AT.WA APR.WA BOS.WA BRS.WA CAV.WA CLC.WA DAT.WA DOM.WA ECH.WA ENA.WA FER.WA FRO.WA GIG.WA GRX.WA HMP.WA INK.WA IZB.WA KGL.WA LWB.WA MAB.WA MCI.WA MED.WA MEX.WA MOC.WA MRC.WA MSZ.WA OPL.WA PEP.WA PLW.WA PXM.WA PZU.WA RFK.WA RLP.WA RWL.WA SCP.WA SEL.WA SNT.WA SNK.WA TOR.WA UNT.WA VOX.WA VRG.WA WLT.WA WWL.WA ZAB.WA ZEP.WA ZUE.WA FSG.WA""".split()
KNF_URL = "https://rss.knf.gov.pl/rss_pub/"
BANKIER_ESPI_RSS = "https://www.bankier.pl/rss/espi.xml"
APP_VERSION = "2.0"

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-muted {color:#777; font-size:0.85rem;}
.badge-buy {background:#d9f7df; padding:4px 10px; border-radius:12px; font-weight:700;}
.badge-hold {background:#fff2b8; padding:4px 10px; border-radius:12px; font-weight:700;}
.badge-sell {background:#ffdada; padding:4px 10px; border-radius:12px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def clean_ticker(x):
    x = str(x).strip().upper()
    if x.endswith(".WA"):
        return x
    return x + ".WA"

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=s.index)
    return (s - med) / (1.4826 * mad)

def clamp(x, lo=0, hi=100):
    return float(np.clip(x, lo, hi)) if pd.notna(x) else np.nan

def signal_label(score):
    if score >= 72:
        return "BUY"
    if score >= 56:
        return "WATCH"
    if score <= 36:
        return "SELL"
    return "HOLD"

# ----------------------------
# Market data
# ----------------------------

@st.cache_data(ttl=900, show_spinner=False)
def download_prices(symbols, period):
    symbols = list(dict.fromkeys(symbols))
    raw = yf.download(
        symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    return raw

def extract_symbol_frame(raw, symbol):
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may return (field, ticker)
        if symbol in raw.columns.get_level_values(-1):
            out = raw.xs(symbol, axis=1, level=-1, drop_level=True).copy()
        elif symbol in raw.columns.get_level_values(0):
            out = raw.xs(symbol, axis=1, level=0, drop_level=True).copy()
        else:
            return pd.DataFrame()
    else:
        out = raw.copy()
    needed = ["Open","High","Low","Close","Volume"]
    if not all(c in out.columns for c in needed):
        return pd.DataFrame()
    return out[needed].dropna(how="all").dropna(subset=["Close"])

def technical_metrics(df):
    c = df["Close"].astype(float)
    v = df["Volume"].fillna(0).astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)

    if len(c) < 30:
        return {}

    s20 = c.rolling(20).mean()
    s50 = c.rolling(50).mean()
    s200 = c.rolling(200).mean()
    r = rsi(c, 14)

    tr = pd.concat([
        h-l,
        (h-c.shift()).abs(),
        (l-c.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    last = safe_float(c.iloc[-1])
    prev = safe_float(c.iloc[-2])
    avg_v20 = safe_float(v.tail(20).mean())
    avg_turn20 = safe_float((c*v).tail(20).mean())

    out = {
        "Kurs": last,
        "1D %": (last/prev-1)*100 if prev else np.nan,
        "5D %": (last/safe_float(c.iloc[-6])-1)*100 if len(c)>6 else np.nan,
        "20D %": (last/safe_float(c.iloc[-21])-1)*100 if len(c)>21 else np.nan,
        "60D %": (last/safe_float(c.iloc[-61])-1)*100 if len(c)>61 else np.nan,
        "RSI14": safe_float(r.iloc[-1]),
        "SMA20": safe_float(s20.iloc[-1]),
        "SMA50": safe_float(s50.iloc[-1]),
        "SMA200": safe_float(s200.iloc[-1]),
        "Vol/20D": safe_float(v.iloc[-1]/avg_v20) if avg_v20 else np.nan,
        "Śr obrót 20D": avg_turn20,
        "ATR14 %": safe_float(atr14.iloc[-1]/last*100) if last else np.nan,
        "High52": safe_float(c.tail(252).max()),
        "Low52": safe_float(c.tail(252).min()),
        "Drawdown52 %": safe_float((last/c.tail(252).max()-1)*100),
        "Dist SMA50 %": safe_float((last/s50.iloc[-1]-1)*100),
        "Dist SMA200 %": safe_float((last/s200.iloc[-1]-1)*100) if pd.notna(s200.iloc[-1]) else np.nan,
    }

    # Trend + momentum + volume subscore (0-100)
    trend = 0
    trend += 25 if last > safe_float(s20.iloc[-1]) else 0
    trend += 25 if last > safe_float(s50.iloc[-1]) else 0
    trend += 20 if pd.notna(s200.iloc[-1]) and last > safe_float(s200.iloc[-1]) else 0
    trend += 15 if safe_float(s20.iloc[-1]) > safe_float(s50.iloc[-1]) else 0
    trend += 15 if pd.notna(s200.iloc[-1]) and safe_float(s50.iloc[-1]) > safe_float(s200.iloc[-1]) else 0

    mom = (
        35 if out["20D %"] > 8 else
        25 if out["20D %"] > 3 else
        15 if out["20D %"] > 0 else
        5 if out["20D %"] > -5 else 0
    )
    rsi_part = (
        100 if 55 <= out["RSI14"] <= 68 else
        75 if 48 <= out["RSI14"] < 55 else
        70 if 68 < out["RSI14"] <= 74 else
        35 if 40 <= out["RSI14"] < 48 else
        20
    )
    volume = 100 if out["Vol/20D"] >= 2 else 85 if out["Vol/20D"] >= 1.3 else 65 if out["Vol/20D"] >= 1.05 else 40

    out["Trend Score"] = trend
    out["Momentum Score"] = 0.6*mom + 0.4*rsi_part
    out["Volume Score"] = volume
    return out

# ----------------------------
# Fundamentals — slower, so only run for technical candidates
# ----------------------------

@st.cache_data(ttl=21600, show_spinner=False)
def fundamental_snapshot(symbols):
    rows = []
    for symbol in symbols:
        rec = {
            "Ticker": symbol.replace(".WA",""),
            "Market Cap": np.nan,
            "P/E": np.nan,
            "P/B": np.nan,
            "EV/EBITDA": np.nan,
            "ROE %": np.nan,
            "Revenue Growth %": np.nan,
            "Profit Margin %": np.nan,
            "FCF TTM": np.nan,
            "CAPEX TTM": np.nan,
            "FCF Yield %": np.nan,
            "Debt/Equity": np.nan,
            "Fundamental Score": np.nan,
        }
        try:
            t = yf.Ticker(symbol)
            info = t.info or {}

            rec["Market Cap"] = safe_float(info.get("marketCap"))
            rec["P/E"] = safe_float(info.get("trailingPE"))
            rec["P/B"] = safe_float(info.get("priceToBook"))
            rec["EV/EBITDA"] = safe_float(info.get("enterpriseToEbitda"))
            rec["ROE %"] = safe_float(info.get("returnOnEquity")) * 100 if info.get("returnOnEquity") is not None else np.nan
            rec["Revenue Growth %"] = safe_float(info.get("revenueGrowth")) * 100 if info.get("revenueGrowth") is not None else np.nan
            rec["Profit Margin %"] = safe_float(info.get("profitMargins")) * 100 if info.get("profitMargins") is not None else np.nan
            rec["Debt/Equity"] = safe_float(info.get("debtToEquity"))

            try:
                cf = t.cashflow
                if cf is not None and not cf.empty:
                    capex_row = None
                    for candidate in ["Capital Expenditure", "Capital Expenditure Reported"]:
                        if candidate in cf.index:
                            capex_row = candidate
                            break
                    if capex_row:
                        vals = pd.to_numeric(cf.loc[capex_row], errors="coerce").dropna()
                        rec["CAPEX TTM"] = safe_float(vals.iloc[:4].sum()) if len(vals) else np.nan

                    fcf_vals = None
                    for candidate in ["Free Cash Flow", "FreeCashFlow"]:
                        if candidate in cf.index:
                            fcf_vals = pd.to_numeric(cf.loc[candidate], errors="coerce").dropna()
                            break
                    if fcf_vals is not None and len(fcf_vals):
                        rec["FCF TTM"] = safe_float(fcf_vals.iloc[:4].sum())
            except Exception:
                pass

            if pd.notna(rec["FCF TTM"]) and pd.notna(rec["Market Cap"]) and rec["Market Cap"] > 0:
                rec["FCF Yield %"] = rec["FCF TTM"] / rec["Market Cap"] * 100

            # Fundamental score from simple, explainable rules.
            score = 50
            if pd.notna(rec["ROE %"]): score += 12 if rec["ROE %"] > 15 else 6 if rec["ROE %"] > 8 else -6 if rec["ROE %"] < 0 else 0
            if pd.notna(rec["Revenue Growth %"]): score += 10 if rec["Revenue Growth %"] > 10 else 5 if rec["Revenue Growth %"] > 3 else -5 if rec["Revenue Growth %"] < -5 else 0
            if pd.notna(rec["FCF Yield %"]): score += 12 if rec["FCF Yield %"] > 8 else 7 if rec["FCF Yield %"] > 4 else -6 if rec["FCF Yield %"] < 0 else 0
            if pd.notna(rec["P/E"]): score += 8 if 0 < rec["P/E"] < 12 else 4 if rec["P/E"] < 20 else -6 if rec["P/E"] > 35 else 0
            if pd.notna(rec["Debt/Equity"]): score += 8 if rec["Debt/Equity"] < 50 else 3 if rec["Debt/Equity"] < 120 else -8 if rec["Debt/Equity"] > 250 else 0
            rec["Fundamental Score"] = clamp(score)
        except Exception:
            pass
        rows.append(rec)
        time.sleep(0.05)
    return pd.DataFrame(rows)

# ----------------------------
# KNF short register
# ----------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_knf_short():
    headers = {
        "User-Agent": "Mozilla/5.0 GPW-DZIDA/2.0",
        "Referer": KNF_URL,
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    session = requests.Session()
    try:
        home = session.get(KNF_URL, headers=headers, timeout=20)
        token = ""
        soup = BeautifulSoup(home.text, "html.parser")
        meta = soup.find("meta", attrs={"name":"_csrf"})
        if meta:
            token = meta.get("content", "")
        if token:
            headers["X-CSRF-TOKEN"] = token
        payload = {
            "request": '{"cmd":"get","language":"pl","search":[],"limit":10000,"offset":0,'
                       '"method":"Default","sort":[{"field":"HOLDER_FULL_NAME","direction":"asc"}],'
                       '"searchLogic":"AND","searchValue":""}'
        }
        resp = session.post(KNF_URL + "JSON", data=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        raw = resp.json()

        # KNF has changed wrappers over time; recursively find list-like records.
        candidates = []
        def walk(obj):
            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        keys = {k.lower() for k in x.keys()}
                        if any("holder" in k for k in keys) and any(("position" in k or "short" in k) for k in keys):
                            candidates.append(x)
                        walk(x)
                    elif isinstance(x, (list,dict)):
                        walk(x)
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
        walk(raw)

        records = candidates[0] if candidates else []
        if not records:
            # fallback: scan every dict in the response for position-ish fields
            stack = [raw]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    lk = {str(k).lower(): k for k in obj.keys()}
                    holder = next((obj[k] for k in obj.keys() if "holder" in str(k).lower()), None)
                    issuer = next((obj[k] for k in obj.keys() if "issuer" in str(k).lower()), None)
                    pos = next((obj[k] for objkey,k in lk.items() if ("netposition" in objkey or "position" in objkey and "date" not in objkey)), None)
                    if holder is not None and issuer is not None and pos is not None:
                        records.append(obj)
                    stack.extend([v for v in obj.values() if isinstance(v,(dict,list))])
                elif isinstance(obj, list):
                    stack.extend(obj)

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(), "KNF returned no parseable records."

        # Normalize names
        def pick(cols):
            for pattern in cols:
                for c in df.columns:
                    if pattern in str(c).lower():
                        return c
            return None

        c_holder = pick(["holder_full_name","holder","short"])
        c_issuer = pick(["issuer_full_name","issuer"])
        c_pos = pick(["netposition","net_position","position"])
        c_date = pick(["positiondate","position_date"])
        c_pub = pick(["publicationdate","publication_date"])

        out = pd.DataFrame({
            "Holder": df[c_holder] if c_holder else "",
            "Issuer": df[c_issuer] if c_issuer else "",
            "Short %": pd.to_numeric(df[c_pos].astype(str).str.replace(",","."), errors="coerce") if c_pos else np.nan,
            "Position Date": pd.to_datetime(df[c_date], errors="coerce", dayfirst=False) if c_date else pd.NaT,
            "Publication Date": pd.to_datetime(df[c_pub], errors="coerce", dayfirst=False) if c_pub else pd.NaT,
        })
        out = out.dropna(subset=["Issuer","Short %"]).copy()
        out["Issuer"] = out["Issuer"].astype(str).str.upper()
        return out, ""
    except Exception as e:
        return pd.DataFrame(), f"KNF fetch error: {e}"

def enrich_short(short_df, universe_df):
    if short_df.empty:
        out = universe_df.copy()
        out["Short Total %"] = np.nan
        out["Short Funds"] = 0
        out["Short Max %"] = np.nan
        out["Short Trend"] = np.nan
        out["Squeeze Score"] = np.nan
        return out

    s = short_df.copy()
    # Match issuer by ticker first; fallback by last tokens.
    s["Ticker"] = ""
    for idx,row in s.iterrows():
        issuer = str(row["Issuer"]).upper()
        matched = None
        for t in universe_df["Ticker"]:
            base = t.upper()
            if base in issuer.replace("S.A.","").replace("SA"," ").replace("."," ").split():
                matched = t
                break
            if issuer.startswith(base + " "):
                matched = t; break
            if (" " + base + " ") in (" " + issuer + " "):
                matched = t; break
        if matched is None:
            known_names = {
                "MODIVO":"CCC", "CD PROJEKT":"CDR", "DINO POLSKA":"DNP",
                "KGHM":"KGH", "JASTRZĘBSKA SPÓŁKA WĘGLOWA":"JSW",
            }
            for name,t in known_names.items():
                if name in issuer:
                    matched = t; break
        s.at[idx,"Ticker"] = matched if matched else ""

    agg = s[s["Ticker"]!=""].groupby("Ticker").agg(
        **{"Short Total %":("Short %","sum"),
           "Short Funds":("Holder","nunique"),
           "Short Max %":("Short %","max")}
    ).reset_index()

    # Trend = latest public position minus previous position for same holder/issuer
    s = s.sort_values(["Holder","Issuer","Position Date"])
    s["prev"] = s.groupby(["Holder","Issuer"])["Short %"].shift(1)
    s["delta"] = s["Short %"] - s["prev"]
    trend = s[s["prev"].notna()].groupby("Ticker")["delta"].sum().rename("Short Trend").reset_index()

    out = universe_df.merge(agg, on="Ticker", how="left").merge(trend, on="Ticker", how="left")
    out["Short Funds"] = out["Short Funds"].fillna(0)

    # squeeze score: high visible short + declining shorts + BUY + volume
    sc = pd.Series(0.0, index=out.index)
    sc += out["Short Total %"].fillna(0).clip(0,5)*10
    sc += np.where(out["Short Trend"].fillna(0)<0, 18, 0)
    sc += np.where(out["D1"]=="BUY", 18, 0)
    sc += np.where(out["Vol/20D"].fillna(0)>=1.25, 12, 0)
    sc += np.where(out["20D %"].fillna(-999)>0, 10, 0)
    sc += np.where(out["Short Funds"].fillna(0)>=3, 8, 0)
    out["Squeeze Score"] = sc.clip(0,100)
    return out

# ----------------------------
# ESPI via Bankier RSS — best effort
# ----------------------------

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_espi_rss(max_items=120):
    try:
        r = requests.get(
            BANKIER_ESPI_RSS,
            headers={"User-Agent":"Mozilla/5.0 GPW-DZIDA/2.0"},
            timeout=20
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")[:max_items]
        rows = []
        for item in items:
            title = item.title.get_text(" ", strip=True) if item.title else ""
            link = item.link.get_text(strip=True) if item.link else ""
            pub = item.pubDate.get_text(strip=True) if item.pubDate else ""
            desc = item.description.get_text(" ", strip=True) if item.description else ""
            rows.append({
                "Date": pd.to_datetime(pub, errors="coerce", utc=True),
                "Title": title, "Link": link, "Description": desc
            })
        return pd.DataFrame(rows), ""
    except Exception as e:
        return pd.DataFrame(), f"ESPI RSS error: {e}"

def insider_score(espi_df, ticker):
    if espi_df.empty:
        return 0, []
    t = ticker.upper().replace(".WA","")
    keys = [t, "TRANSAKCJ", "ART. 19", "AKCJ"]
    matches = []
    for _,r in espi_df.iterrows():
        title = str(r["Title"]).upper()
        if t in title and ("TRANSAKCJ" in title or "NABYCIE AKC" in title or "ZBYCIE AKC" in title):
            matches.append(r)
    # Prefer purchases over sales in scoring.
    buy = sum(1 for r in matches if re.search(r" NABY|ZAKUP|NABYCIE|KUP", str(r["Title"]).upper()))
    sell = sum(1 for r in matches if re.search(r" ZBY|SPRZED|ZBYCIE", str(r["Title"]).upper()))
    score = 50 + buy*18 - sell*18
    return int(np.clip(score,0,100)), matches[:8]

# ----------------------------
# Build universe
# ----------------------------

st.sidebar.title("⚙️ GPW DZIDA Advanced")
uploaded = st.sidebar.file_uploader("Własna lista spółek CSV (ticker)", type="csv")
period = st.sidebar.selectbox("Historia cen", ["6mo","1y","2y","5y"], index=1)
min_turnover = st.sidebar.number_input("Min. średni obrót 20D [PLN]", min_value=0, value=750_000, step=250_000)
top_n = st.sidebar.slider("TOP N", 5, 30, 10)
do_fundamentals = st.sidebar.checkbox("Pobieraj fundamenty + CAPEX", value=True)
use_live_knf = st.sidebar.checkbox("Automatyczny KNF short register", value=True)
use_live_espi = st.sidebar.checkbox("Automatyczny ESPI/insider RSS", value=True)

if uploaded:
    u = pd.read_csv(uploaded)
    if "ticker" not in u.columns:
        st.error("CSV musi zawierać kolumnę 'ticker'.")
        st.stop()
    symbols = [clean_ticker(x) for x in u["ticker"].dropna()]
else:
    symbols = DEFAULT_SYMBOLS

st.sidebar.caption("Domyślna lista jest przykładową listą płynnych walorów. Pełną listę GPW możesz podmienić przez CSV.")

with st.spinner(f"Pobieram ceny: {len(symbols)} spółek..."):
    prices = download_prices(tuple(symbols), period)

tech_rows = []
for sym in symbols:
    f = extract_symbol_frame(prices, sym)
    m = technical_metrics(f)
    if m:
        m["Ticker"] = sym.replace(".WA","")
        tech_rows.append(m)

df = pd.DataFrame(tech_rows)
if df.empty:
    st.error("Nie udało się pobrać danych cenowych.")
    st.stop()

# relative strength vs WIG if available
bench = extract_symbol_frame(download_prices(("WIG.WA",), period), "WIG.WA")
if not bench.empty:
    bc = bench["Close"]
    wig_20 = (safe_float(bc.iloc[-1])/safe_float(bc.iloc[-21])-1)*100 if len(bc)>21 else np.nan
    df["RS vs WIG 20D"] = df["20D %"] - wig_20
else:
    df["RS vs WIG 20D"] = np.nan

# D1 technical composite
df["Technical Score"] = (
    df["Trend Score"]*0.35 +
    df["Momentum Score"]*0.35 +
    df["Volume Score"]*0.15 +
    df["ATR14 %"].rank(pct=True).mul(100).fillna(50)*0.15
)
# Liquidity
df["Liquidity Score"] = (np.log1p(df["Śr obrót 20D"].clip(lower=0)).rank(pct=True)*100).fillna(0)

df["D1"] = np.where(df["Technical Score"]>=72, "BUY",
             np.where(df["Technical Score"]>=55,"WATCH",
             np.where(df["Technical Score"]<=36,"SELL","HOLD")))

# Liquidity filter
liquid = df[df["Śr obrót 20D"]>=min_turnover].copy()

# Fundamentals only for top technical candidates
if do_fundamentals and not liquid.empty:
    candidates = liquid.sort_values("Technical Score", ascending=False).head(min(80, len(liquid)))
    fund = fundamental_snapshot([clean_ticker(x) for x in candidates["Ticker"]])
    liquid = liquid.merge(fund, on="Ticker", how="left")
else:
    liquid["Fundamental Score"] = 50
    for c in ["Market Cap","P/E","P/B","EV/EBITDA","ROE %","Revenue Growth %","Profit Margin %","FCF TTM","CAPEX TTM","FCF Yield %","Debt/Equity"]:
        liquid[c] = np.nan

# defaults/fallbacks
liquid["Fundamental Score"] = liquid["Fundamental Score"].fillna(50)

# live KNF
knf_message = ""
if use_live_knf:
    knf_df, knf_message = fetch_knf_short()
else:
    knf_df = pd.DataFrame()

universe_for_short = liquid[["Ticker","D1","Vol/20D","20D %"]].copy()
if use_live_knf:
    liquid = enrich_short(knf_df, universe_for_short)
else:
    liquid["Short Total %"] = np.nan
    liquid["Short Funds"] = 0
    liquid["Short Max %"] = np.nan
    liquid["Short Trend"] = np.nan
    liquid["Squeeze Score"] = np.nan

# insider feed
espi_df = pd.DataFrame()
espi_err = ""
if use_live_espi:
    espi_df, espi_err = fetch_espi_rss()

# Final score:
# technical 45%, fundamentals 25%, liquidity 10%, squeeze/positioning 20%
liquid["Insider Score"] = 50.0
if not espi_df.empty:
    for i,row in liquid.iterrows():
        sc,_ = insider_score(espi_df, row["Ticker"])
        liquid.at[i,"Insider Score"] = sc

liquid["Positioning Score"] = liquid["Squeeze Score"].fillna(50)
liquid["Final Score"] = (
    liquid["Technical Score"]*0.45 +
    liquid["Fundamental Score"]*0.25 +
    liquid["Liquidity Score"]*0.10 +
    liquid["Positioning Score"]*0.15 +
    liquid["Insider Score"]*0.05
).clip(0,100)
liquid["Signal"] = liquid["Final Score"].map(signal_label)

# ----------------------------
# DASHBOARD
# ----------------------------

st.title("📈 GPW DZIDA — Advanced Scanner")
st.caption(f"Wersja {APP_VERSION} · odświeżanie cache: ceny 15 min, KNF 30 min, ESPI 20 min.")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Spółki w skanerze", len(df))
k2.metric("Płynne", len(liquid))
k3.metric("D1 BUY", int((liquid["D1"]=="BUY").sum()))
k4.metric("TOP ≥72", int((liquid["Final Score"]>=72).sum()))
k5.metric("Widoczne shorty", int(liquid["Short Total %"].notna().sum()))

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🏠 Rynek", "🚀 TOP okazje", "💥 Short squeeze", "📊 Fundamenty", "🔎 Spółka"
])

with tab1:
    st.subheader("Szerokość rynku")
    c1,c2 = st.columns(2)
    with c1:
        pct_up = (liquid["1D %"]>0).mean()*100 if len(liquid) else np.nan
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct_up,
            number={"suffix":"%"},
            title={"text":"Spółki rosnące dziś"},
            gauge={"axis":{"range":[0,100]}},
        ))
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Śr. zmiana 1D", f"{liquid['1D %'].mean():.2f}%")
        st.metric("Śr. zmiana 20D", f"{liquid['20D %'].mean():.2f}%")
        st.metric("Śr. RSI", f"{liquid['RSI14'].mean():.1f}")
        st.metric("Śr. obrót 20D", f"{liquid['Śr obrót 20D'].mean()/1e6:.1f} mln PLN")

    st.subheader("Największe ruchy")
    a,b = st.columns(2)
    with a:
        st.markdown("**▲ Wzrosty**")
        cols=["Ticker","Kurs","1D %","5D %","20D %","D1","Final Score"]
        st.dataframe(liquid.nlargest(10,"1D %")[cols].round(2), use_container_width=True, hide_index=True)
    with b:
        st.markdown("**▼ Spadki**")
        st.dataframe(liquid.nsmallest(10,"1D %")[cols].round(2), use_container_width=True, hide_index=True)

    st.subheader("Rozkład sygnałów")
    sig = liquid["Signal"].value_counts().rename_axis("Signal").reset_index(name="Liczba")
    fig = px.bar(sig, x="Signal", y="Liczba", text="Liczba")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("TOP GPW — Final Score")
    order = liquid.sort_values(["Final Score","Technical Score"], ascending=False).head(top_n).copy()
    cols = ["Ticker","Kurs","Signal","D1","Final Score","Technical Score","Fundamental Score",
            "Liquidity Score","Positioning Score","Insider Score","20D %","RSI14","Vol/20D",
            "Śr obrót 20D","Upside do 52W high %","CAPEX TTM"]
    order["Upside do 52W high %"] = (order["High52"]/order["Kurs"]-1)*100
    cols = [c for c in cols if c in order.columns]
    st.dataframe(order[cols].round(2), use_container_width=True, hide_index=True)

    st.subheader("Scoring — gdzie powstaje wynik?")
    if not order.empty:
        bar = order.head(min(10,len(order))).copy().sort_values("Final Score")
        fig = go.Figure()
        for c in ["Technical Score","Fundamental Score","Liquidity Score","Positioning Score","Insider Score"]:
            fig.add_bar(y=bar["Ticker"], x=bar[c], name=c, orientation="h")
        fig.update_layout(
            barmode="stack", height=460, xaxis_title="Punkty", yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("💥 Short squeeze radar")
    if liquid["Short Total %"].notna().sum()==0:
        st.warning("KNF nie zwrócił danych w tej sesji. Włącz automatyczny KNF lub sprawdź komunikat poniżej.")
        if knf_message: st.caption(knf_message)
    else:
        sq = liquid[liquid["Short Total %"].notna()].sort_values(
            ["Squeeze Score","Short Total %"], ascending=False
        ).head(top_n)
        cols=["Ticker","Signal","D1","Squeeze Score","Short Total %","Short Max %",
              "Short Funds","Short Trend","1D %","20D %","Vol/20D","Final Score"]
        st.dataframe(sq[cols].round(2), use_container_width=True, hide_index=True)

        if not sq.empty:
            fig = px.scatter(
                liquid[liquid["Short Total %"].notna()],
                x="Short Total %", y="20D %",
                size="Short Funds", color="Squeeze Score",
                hover_name="Ticker",
                labels={"Short Total %":"Widoczny short % kapitału", "20D %":"Zmiana 20D %"}
            )
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Interpretacja: wysoki short + dodatnie momentum + malejący Short Trend + "
            "podwyższony wolumen tworzą charakterystyczny układ potencjalnego short squeeze. "
            "Rejestr KNF pokazuje tylko publiczne pozycje od 0,5% kapitału; brak wpisu nie oznacza zerowego short interest."
        )

with tab4:
    st.subheader("Fundamenty + CAPEX")
    fcols=["Ticker","Market Cap","P/E","P/B","EV/EBITDA","ROE %","Revenue Growth %",
           "Profit Margin %","FCF TTM","CAPEX TTM","FCF Yield %","Debt/Equity","Fundamental Score"]
    ff = liquid.sort_values("Fundamental Score", ascending=False)
    st.dataframe(ff[[c for c in fcols if c in ff.columns]].round(2),
                 use_container_width=True, hide_index=True)

    st.subheader("CAPEX / FCF")
    tmp = ff[["Ticker","FCF TTM","CAPEX TTM"]].dropna().head(20)
    if not tmp.empty:
        melted = tmp.melt(id_vars="Ticker", var_name="Miara", value_name="PLN")
        fig = px.bar(melted, x="Ticker", y="PLN", color="Miara", barmode="group")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    if do_fundamentals:
        st.caption("CAPEX jest pobierany z cash flow statement; wartości zależą od dostępności i jakości danych Yahoo Finance.")

with tab5:
    st.subheader("🔎 Szczegółowa analiza spółki")
    choices = sorted(liquid["Ticker"].dropna().unique())
    selected = st.selectbox("Wybierz ticker", choices if choices else ["—"])
    row = liquid[liquid["Ticker"]==selected].iloc[0] if selected in choices else None

    if row is not None:
        a,b,c,d = st.columns(4)
        a.metric("Kurs", f"{row['Kurs']:.2f}")
        b.metric("Final Score", f"{row['Final Score']:.1f}/100")
        c.metric("D1", row["D1"])
        d.metric("Squeeze", f"{row['Squeeze Score']:.1f}" if pd.notna(row["Squeeze Score"]) else "brak")

        st.markdown("### Profil sygnału")
        prof = pd.DataFrame({
            "Obszar":["Technika","Fundamenty","Płynność","Short/Squeeze","Insider"],
            "Score":[row["Technical Score"],row["Fundamental Score"],row["Liquidity Score"],
                     row["Positioning Score"],row["Insider Score"]]
        })
        fig = px.bar(prof, x="Score", y="Obszar", orientation="h", text="Score")
        fig.update_layout(height=300, xaxis={"range":[0,100]})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Kluczowe parametry")
        params = {
            "1D %": row["1D %"], "5D %": row["5D %"], "20D %": row["20D %"],
            "RSI14": row["RSI14"], "Vol/20D": row["Vol/20D"], "ATR14 %": row["ATR14 %"],
            "RS vs WIG 20D": row["RS vs WIG 20D"],
            "Śr obrót 20D": row["Śr obrót 20D"],
            "P/E": row.get("P/E",np.nan), "ROE %": row.get("ROE %",np.nan),
            "FCF Yield %": row.get("FCF Yield %",np.nan),
            "CAPEX TTM": row.get("CAPEX TTM",np.nan),
            "Short Total %": row.get("Short Total %",np.nan),
            "Short Trend": row.get("Short Trend",np.nan),
        }
        st.dataframe(pd.DataFrame([params]).T.rename(columns={0:"Wartość"}).round(2), use_container_width=True)

        hist = extract_symbol_frame(prices, clean_ticker(selected))
        if not hist.empty:
            st.markdown("### Wykres kursu")
            c = hist["Close"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=c, name="Close"))
            fig.add_trace(go.Scatter(x=hist.index, y=c.rolling(20).mean(), name="SMA20"))
            fig.add_trace(go.Scatter(x=hist.index, y=c.rolling(50).mean(), name="SMA50"))
            if len(c)>=200:
                fig.add_trace(go.Scatter(x=hist.index, y=c.rolling(200).mean(), name="SMA200"))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        if use_live_espi and not espi_df.empty:
            sc, items = insider_score(espi_df, selected)
            st.markdown(f"### ESPI / insider score: **{sc}/100**")
            if items:
                st.dataframe(pd.DataFrame(items)[["Date","Title","Link"]], use_container_width=True, hide_index=True)
            else:
                st.caption("Brak rozpoznanych najnowszych komunikatów transakcyjnych dla tego tickera.")

# ----------------------------
# Diagnostics / data sources
# ----------------------------

with st.expander("ℹ️ Status źródeł danych"):
    st.write("Yahoo Finance / ceny:", "OK" if not df.empty else "BRAK")
    st.write("KNF Short Register:", "OK" if not knf_df.empty else "BRAK / nieprzetworzone")
    st.write("ESPI RSS:", "OK" if not espi_df.empty else "BRAK / niedostępne")
    if knf_message: st.caption(knf_message)
    if espi_err: st.caption(espi_err)

st.divider()
st.caption(
    "GPW DZIDA nie jest rekomendacją inwestycyjną. Score jest autorskim rankingiem ilościowym. "
    "KNF publikuje tylko publiczne znaczące pozycje krótkie; fundamenty i CAPEX zależą od danych dostawcy."
)
