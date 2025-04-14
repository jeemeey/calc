# Smart Compound + Recovery + Sharing
import streamlit as st
from PIL import Image
import pandas as pd
from io import BytesIO
import smtplib
from email.message import EmailMessage
import os
import datetime
from PIL import Image
import re
import pytesseract
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import base64



# === Compound Growth ===
def compound_growth(principal, rate, days, frequency, withdraw):
    freq = {"Daily": 1, "Weekly": 7, "Monthly": 30}[frequency]
    data = []
    balance = principal
    withdrawn = 0

    for day in range(1, days + 1):
        if day % freq == 0:
            profit = balance * (rate / 100)
            if withdraw:
                withdrawn += profit
            else:
                balance += profit
        else:
            profit = 0

        data.append({
            "Day": day,
            "Profit": round(profit, 2),
            "Balance": round(balance, 2),
            "Withdrawn": round(withdrawn, 2)
        })

    return pd.DataFrame(data), withdrawn, balance

# === Profit-Only Withdrawal ===
def profit_only_withdrawal(initial, rate, days, fallback_ratio=2/3):
    balance = initial
    withdrawn = 0
    goal = initial * fallback_ratio
    data = []

    for day in range(1, days + 1):
        profit = balance * (rate / 100)
        take = min(profit, initial - withdrawn)
        balance += profit - take
        withdrawn += take

        data.append({
            "Day": day,
            "Daily Profit": round(profit, 2),
            "Withdrawn": round(take, 2),
            "Remaining": round(max(0, initial - withdrawn), 2),
            "Balance": round(balance, 2)
        })

        if withdrawn >= initial:
            break

    status = (
        f"✅ Full Recovery in {day} days"
        if withdrawn >= initial else
        f"⚠️ Partial Recovery: {withdrawn:.2f} ({(withdrawn / initial) * 100:.1f}%)"
    )

    return pd.DataFrame(data), withdrawn, balance, status

# === Shared Strategy ===
def shared_strategy(initial, schedule, split, people):
    balance = initial
    recovered = 0
    day = 1
    goal = initial
    rows = []

    while recovered < goal:
        rate = next((s['rate'] for s in schedule if s['from'] <= day <= s['to']), schedule[-1]['rate'])
        profit = balance * (rate / 100)
        rec_part = profit * split
        shared = profit * (1 - split)
        rec_now = min(rec_part, goal - recovered)
        recovered += rec_now
        balance += profit - rec_now

        row = {
            "Day": day,
            "Rate %": rate,
            "Profit": round(profit, 2),
            "Recovered": round(rec_now, 2),
            "Remaining to Recover": round(goal - recovered, 2),
            "Balance": round(balance, 2)
        }
        for i in range(1, people + 1):
            row[f"Person {i} Share"] = round(shared / people, 2)

        rows.append(row)
        day += 1

    return pd.DataFrame(rows), recovered, day

# === Export Excel ===
def convert_df_to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return out.getvalue()

# === Save History ===
def save_to_history(entry):
    df = pd.DataFrame([entry])
    if os.path.exists("history.csv"):
        df.to_csv("history.csv", mode="a", index=False, header=False)
    else:
        df.to_csv("history.csv", index=False)

# === Send Email ===
def send_email(to, subject, body, attachment, filename):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to
    msg.set_content(body)
    msg.add_attachment(attachment, maintype="application", subtype="octet-stream", filename=filename)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)

# === Streamlit App ===
st.set_page_config(page_title="Smart Investment App", layout="centered")
st.title("💹 Smart Compound + Recovery + Sharing")
tabs = st.tabs([
    "📈 Compound Calculator",
    "💸 Withdrawal Simulator",
    "🤝 Shared Investment Plan",
    "Range Summary Analysis",
    "📊 Jemey Real-Time Dashboard"
])


# === CONFIG ===
FULL_JEMEY_PASSWORD = "ahmedelite"  # Set your unlock password
LOGO_PATH = "C:/Users/ahmed/Downloads/jemeyai/logos/jemey.png"  # Your logo path
MODE_FILE = "jemey_mode.txt"  # For saving mode across sessions

# === Load saved mode if exists ===
if "jemey_mode" not in st.session_state:
    if os.path.exists(MODE_FILE):
        with open(MODE_FILE, "r") as f:
            st.session_state.jemey_mode = f.read().strip()
    else:
        st.session_state.jemey_mode = "Normal"

# === Convert image to base64 ===
def get_base64_image(LOGO_PATH):
    with open(LOGO_PATH, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# === Render Jemey Sidebar Logo & Title ===
def render_logo_sidebar():
    img_data = get_base64_image(LOGO_PATH)
    mode = st.session_state.get("jemey_mode", "Normal")

    if mode == "Full":
        jemey_title = "💠 JEMEY Engine"
        jemey_color = "#00f0ff"
    else:
        jemey_title = "🤖 JEMEY Normal"
        jemey_color = "#888888"

    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_data}" width="150" style="margin-bottom: 10px;">
            <h3 style="color:{jemey_color}; margin-top: 5px;">{jemey_title}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# === Unlock logic ===
if st.session_state.jemey_mode != "Full":
    st.sidebar.markdown("### 🔐 Jemey Access")
    auth_input = st.sidebar.text_input("Enter Unlock Key", type="password")
    if st.sidebar.button("🔓 Unlock Full Jemey"):
        if auth_input == FULL_JEMEY_PASSWORD:
            st.session_state.jemey_mode = "Full"
            with open(MODE_FILE, "w") as f:
                f.write("Full")
            st.sidebar.success("✅ Jemey Engine Activated")
            st.rerun()

        else:
            st.sidebar.error("❌ Wrong password. Try again.")
else:
    # Show "Lock" button if already in Full mode
    if st.sidebar.button("🔒 Lock Jemey (Back to Normal)"):
        st.session_state.jemey_mode = "Normal"
        with open(MODE_FILE, "w") as f:
            f.write("Normal")
        st.sidebar.info("🔐 Jemey is now in Normal Mode")
        st.rerun()


# === Show logo after logic ===
render_logo_sidebar()



# === Tab 1 ===
with tabs[0]:
    st.subheader("📈 Compound Interest Calculator")
    col1, col2 = st.columns(2)
    with col1:
        principal = st.number_input("Initial Investment", 500.0, step=50.0)
        rate = st.number_input("Interest Rate (%)", 2.217 , step=0.1)
        freq = st.selectbox("Compounding Frequency", ["Daily", "Weekly", "Monthly"])
    with col2:
        days = st.number_input("Days", 5, step=5)
        withdraw = st.checkbox("Withdraw profits?")
        currency = st.text_input("Currency Symbol", "$")

    auto_save = st.checkbox("Auto-save to history.csv")
    email = st.text_input("📧 Send result to email")

    if st.button("Calculate Compound Growth"):
        df, withdrawn, final = compound_growth(principal, rate, days, freq, withdraw)
        total_profit = df["Profit"].sum()

        st.success(f"💰 Total Profit: {currency}{total_profit:,.2f}")
        st.success(f"🏦 Final Balance: {currency}{final:,.2f}")
        if withdraw:
            st.info(f"💸 Withdrawn: {currency}{withdrawn:,.2f}")

        st.line_chart(df.set_index("Day")[["Balance", "Withdrawn"]])
        st.dataframe(df)
        excel = convert_df_to_excel(df)
        st.download_button("⬇️ Excel", data=excel, file_name="compound.xlsx")

        if auto_save:
            save_to_history({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Initial": principal,
                "Rate %": rate,
                "Days": days,
                "Profit": total_profit,
                "Final": final,
                "Withdrawn": withdrawn,
                "Mode": withdraw,
                "Frequency": freq
            })

        if email:
            try:
                send_email(email, "Compound Report", f"Final: {final:.2f}, Profit: {total_profit:.2f}", excel, "compound.xlsx")
                st.success("📧 Email sent!")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")

# === Tab 2 ===

with tabs[1]:
    st.subheader("💸 Daily Withdrawal Simulator")
    col1, col2 = st.columns(2)
    with col1:
        w_init = st.number_input("Initial Investment", 500.0, step=50.0, key="w1")
        w_days = st.number_input("Simulate for Days", 5, key="w2")
    with col2:
        w_rate = st.number_input("Daily Rate (%)", 2.217 , key="w3")
        fallback_ratio = st.number_input("Fallback Recovery Ratio (e.g. 0.67)", value=0.67, min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Simulate Withdrawal"):
        w_df, w_total, w_balance, status = profit_only_withdrawal(w_init, w_rate, w_days, fallback_ratio)
        st.info(status)
        st.success(f"💰 Withdrawn: {w_total:,.2f}")
        st.success(f"🏦 Final Balance: {w_balance:,.2f}")
        st.line_chart(w_df.set_index("Day")[["Balance", "Withdrawn"]])
        st.dataframe(w_df)


# === Tab 3 ===

with tabs[2]:
    st.subheader("🤝 Shared Strategy: Custom Profit Split & Fast Recovery")

    col1, col2 = st.columns(2)
    with col1:
        invest = st.number_input("Initial Investment", value=500.0, step=50.0)
        schedule_str = st.text_area("Gain Schedule (e.g. 1-10:3.3,11-20:2.2)", "1-10:3.3,11-20:2.2,21-30:4.4")
    with col2:
        split_ratio = st.number_input("Recovery Split Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        people_count = st.number_input("Number of People Sharing Profit", min_value=1, value=2, step=1)

    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    names, ratios = [], []
    st.markdown("### 🔧 Custom Profit Split Ratio")
    for i in range(1, people_count + 1):
        col1, col2 = st.columns([2, 1])
        with col1:
            name = st.text_input(f"Name of Person {i}", value=f"Person {i}", key=f"name{i}")
        with col2:
            ratio = st.number_input(f"{name}'s Share (%)", min_value=0.0, max_value=100.0,
                                    value=round(100 / people_count, 2), key=f"ratio{i}")
        names.append(name)
        ratios.append(ratio)

    total_ratio = sum(ratios)
    if abs(total_ratio - 100.0) > 0.01:
        st.error(f"⚠️ Total ratio must equal 100%. Current total: {total_ratio:.2f}%")
    else:
        if st.button("▶️ Run Custom Sharing Strategy"):
            rows = []
            balance = invest
            recovered = 0
            goal = invest
            day = 1
            max_days = 1000

            while recovered < goal and day <= max_days:
                rate = next((s['rate'] for s in schedule if s['from'] <= day <= s['to']), schedule[-1]['rate'])
                profit = balance * (rate / 100)
                recovery_part = profit * split_ratio
                shared_part = profit * (1 - split_ratio)

                recovered_today = min(recovery_part, goal - recovered)
                recovered += recovered_today
                balance += profit - recovered_today

                row = {
                    "Day": day,
                    "Rate %": rate,
                    "Profit": round(profit, 2),
                    "Recovered": round(recovered_today, 2),
                    "Balance": round(balance, 2),
                    "Remaining to Recover": round(goal - recovered, 2)
                }

                for name, ratio in zip(names, ratios):
                    row[f"{name} Share"] = round(shared_part * (ratio / 100), 2)

                rows.append(row)
                day += 1

            df = pd.DataFrame(rows)
            share_cols = [f"{name} Share" for name in names]

            st.session_state["df"] = df
            st.session_state["share_cols"] = share_cols

            if df.empty:
                st.error("⚠️ Recovery failed: no data generated.")
            else:
                st.success(f"✅ Full Recovery Achieved in {day - 1} days")
                st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
                st.dataframe(df)

              
# === Tab 4 ===

with tabs[3]:
    st.subheader("📊 Range Summary Analysis")

    if "df" in st.session_state and not st.session_state["df"].empty:
        df = st.session_state["df"]
        share_cols = st.session_state["share_cols"]

        max_day = int(df["Day"].max())
        from_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="range_from_day")
        to_day = st.number_input("End Day", min_value=from_day, max_value=max_day, value=from_day + 5, key="range_to_day")

        selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="range_cols")

        if st.button("✅ Analyze Range Summary"):
            try:
                range_df = df[(df["Day"] >= from_day) & (df["Day"] <= to_day)]

                if range_df.empty:
                    st.warning("⚠️ No data in selected range.")
                elif not selected_cols:
                    st.warning("⚠️ Please select people to analyze.")
                else:
                    selected_range = range_df[selected_cols]
                    st.subheader("📊 Totals")
                    st.dataframe(selected_range.sum().round(2).to_frame().T)

                    st.subheader("📈 Averages")
                    st.dataframe(selected_range.mean().round(2).to_frame().T)

                    st.subheader("🔺 Max")
                    st.dataframe(selected_range.max().round(2).to_frame().T)

                    st.subheader("🔻 Min")
                    st.dataframe(selected_range.min().round(2).to_frame().T)

                    st.line_chart(selected_range)

            except Exception as e:
                st.error(f"💥 Error during analysis: {e}")
    else:
        st.info("ℹ️ Run the sharing strategy in Tab 3 first to generate data.")


# === Tab 5 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)

with tabs[4]:
    st.title("📈 Jemey Live Market Dashboard")

    st.header("📊 Market Settings")
    asset_type = st.selectbox("Select Asset Type", ["Crypto", "Stock"])

    default_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"] if asset_type == "Crypto" else ["AAPL", "TSLA", "LE"]
    symbol = st.selectbox("Select Symbol", default_symbols)
    custom_symbol = st.text_input("Or enter custom symbol", "")
    active_symbol = custom_symbol if custom_symbol else symbol

    lookback = st.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

    @st.cache_data
    def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
        df = yf.download(tickers=symbol, period=period, interval=interval)
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        rename_map = {col: col.replace(f" {symbol}", "") for col in df.columns if f" {symbol}" in col}
        df.rename(columns=rename_map, inplace=True)
        return df

    try:
        df = fetch_market_data(active_symbol, period=lookback)

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            df.rename(columns={
                'Datetime': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
            }, inplace=True)

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in market data.")

        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
         
         # 🧮 Live Indicator Values
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        close = df['Close'].iloc[-1]
        bb_pos = (close - df['Low'].min()) / (df['High'].max() - df['Low'].min()) * 100
        atr_pct = (atr / close) * 100


        # Sample values to simulate the layout
        rsi = 54.58
        macd = -1036.76
        bb_position = "68.9%"
        volatility = "4.99%"

        ema20_50 = "Bearish"
        ema50_100 = "Bearish"
        ema100_200 = "Bullish"
        overall_trend = "Bearish"

        # Display section header
        st.markdown("## 📊 Technical Analysis Summary of `BTC-USD`")

        # Row 1: RSI, MACD, BB Position, Volatility (4 cards side by side)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### RSI")
            st.markdown("**Value:** 54.58")
            st.markdown("Normal Trading Range (RSI 30-70)")
            st.markdown("- Normal market conditions with balanced buying/selling")
            st.markdown("- Focus on trend following strategies")
            st.markdown("- Look for RSI direction and momentum")
            st.markdown("- Use other indicators for trade signals")

        with col2:
            st.markdown("### MACD")
            st.markdown("**Value:** -1036.76")
            st.markdown("Bullish MACD Cross")
            st.markdown("- Momentum is shifting bullish")
            st.markdown("- Consider long positions with positive histogram")
            st.markdown("- Stronger signal if crossover occurs below zero")
            st.markdown("- Use RSI and BB confirmation for better entries")

        with col3:
            st.markdown("### BB Position")
            st.markdown("**Value:** 68.9%")
            st.markdown("Price Within Bands")
            st.markdown("- Normal price movement within statistical range")
            st.markdown("- Watch for potential breakout if bands narrow")
            st.markdown("- Use other indicators to determine direction")
            st.markdown("- Consider setting alerts for band breaks")

        with col4:
            st.markdown("### Volatility (ATR)")
            st.markdown("**Value:** 4.99%")
            st.markdown("High Volatility (>3%)")
            st.markdown("- Large price movements are common")
            st.markdown("- Widen stop losses to account for swings")
            st.markdown("- Reduce position sizes to manage risk")
            st.markdown("- Consider staying out if too volatile")

        # Display section header
        st.markdown("## 📈 Moving Average Analysis")

        # Row 2: EMA Crosses
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.markdown("### EMA20/50 Cross")
            st.markdown("**Trend:** Bearish")
            st.markdown("Death Cross (Bearish)")
            st.markdown("- Shorter EMA crossed below longer EMA")
            st.markdown("- Strong bearish momentum signal")
            st.markdown("- Consider short positions with confirmation")
            st.markdown("- Use resistance levels for stop placement")

        with col6:
            st.markdown("### EMA50/100 Cross")
            st.markdown("**Trend:** Bearish")
            st.markdown("Death Cross (Bearish)")
            st.markdown("- Shorter EMA crossed below longer EMA")
            st.markdown("- Strong bearish momentum signal")
            st.markdown("- Consider short positions with confirmation")
            st.markdown("- Use resistance levels for stop placement")

        with col7:
            st.markdown("### EMA100/200 Cross")
            st.markdown("**Trend:** Bullish")
            st.markdown("Golden Cross (Bullish)")
            st.markdown("- Shorter EMA crossed above longer EMA")
            st.markdown("- Strong bullish momentum signal")
            st.markdown("- Consider long positions with confirmation")
            st.markdown("- Use support levels for stop placement")

        with col8:
            st.markdown("### Overall EMA Trend")
            st.markdown("**Trend:** Bearish")
            st.markdown("Strong Bearish Trend")
            st.markdown("- Multiple EMA crosses confirm downtrend")
            st.markdown("- Higher timeframe momentum is negative")
            st.markdown("- Look for rallies to EMAs as resistance")
            st.markdown("- Consider longer-term short positions")

        # 📊 Price + EMA Chart
        st.subheader(f"📈 Price Chart + EMA for {active_symbol}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Candles'))
        for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 📉 RSI
        st.subheader("📉 RSI")
        fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # 📈 MACD
        st.subheader("📈 MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
        fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
        st.plotly_chart(fig_macd, use_container_width=True)

        # 📊 ATR
        st.subheader("📊 ATR - Volatility")
        fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
        st.plotly_chart(fig_atr, use_container_width=True)

        # 💡 Risk Guidelines
        with st.expander("💡 Risk Management Guidelines"):
            st.markdown(""""
         **🔢 Position Sizing**
         - Risk 1–2% per trade
         - Adjust for volatility

            **🛑 Stop Loss Strategy**
            - Place SL outside noise
            - Use ATR or support/resistance

            **📉 Market Correlation**
            - Reduce exposure during high correlation
            - Diversify assets

            **🧠 Trade Management**
            - Use risk/reward of 1:2 or better
            - Never average down losers
            """)

        st.subheader("📂 Raw Market Data")
        st.dataframe(df)

        st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

    except Exception as e:
        st.error(f"⚠️ Failed to fetch market data: {e}")

    st.markdown("---")
    st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by JEMEY AI Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
# Smart Compound + Recovery + Sharing
import streamlit as st
from PIL import Image
import pandas as pd
from io import BytesIO
import smtplib
from email.message import EmailMessage
import os
import datetime
from PIL import Image
import re
import pytesseract
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import base64



# === Compound Growth ===
def compound_growth(principal, rate, days, frequency, withdraw):
    freq = {"Daily": 1, "Weekly": 7, "Monthly": 30}[frequency]
    data = []
    balance = principal
    withdrawn = 0

    for day in range(1, days + 1):
        if day % freq == 0:
            profit = balance * (rate / 100)
            if withdraw:
                withdrawn += profit
            else:
                balance += profit
        else:
            profit = 0

        data.append({
            "Day": day,
            "Profit": round(profit, 2),
            "Balance": round(balance, 2),
            "Withdrawn": round(withdrawn, 2)
        })

    return pd.DataFrame(data), withdrawn, balance

# === Profit-Only Withdrawal ===
def profit_only_withdrawal(initial, rate, days, fallback_ratio=2/3):
    balance = initial
    withdrawn = 0
    goal = initial * fallback_ratio
    data = []

    for day in range(1, days + 1):
        profit = balance * (rate / 100)
        take = min(profit, initial - withdrawn)
        balance += profit - take
        withdrawn += take

        data.append({
            "Day": day,
            "Daily Profit": round(profit, 2),
            "Withdrawn": round(take, 2),
            "Remaining": round(max(0, initial - withdrawn), 2),
            "Balance": round(balance, 2)
        })

        if withdrawn >= initial:
            break

    status = (
        f"✅ Full Recovery in {day} days"
        if withdrawn >= initial else
        f"⚠️ Partial Recovery: {withdrawn:.2f} ({(withdrawn / initial) * 100:.1f}%)"
    )

    return pd.DataFrame(data), withdrawn, balance, status

# === Shared Strategy ===
def shared_strategy(initial, schedule, split, people):
    balance = initial
    recovered = 0
    day = 1
    goal = initial
    rows = []

    while recovered < goal:
        rate = next((s['rate'] for s in schedule if s['from'] <= day <= s['to']), schedule[-1]['rate'])
        profit = balance * (rate / 100)
        rec_part = profit * split
        shared = profit * (1 - split)
        rec_now = min(rec_part, goal - recovered)
        recovered += rec_now
        balance += profit - rec_now

        row = {
            "Day": day,
            "Rate %": rate,
            "Profit": round(profit, 2),
            "Recovered": round(rec_now, 2),
            "Remaining to Recover": round(goal - recovered, 2),
            "Balance": round(balance, 2)
        }
        for i in range(1, people + 1):
            row[f"Person {i} Share"] = round(shared / people, 2)

        rows.append(row)
        day += 1

    return pd.DataFrame(rows), recovered, day

# === Export Excel ===
def convert_df_to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return out.getvalue()

# === Save History ===
def save_to_history(entry):
    df = pd.DataFrame([entry])
    if os.path.exists("history.csv"):
        df.to_csv("history.csv", mode="a", index=False, header=False)
    else:
        df.to_csv("history.csv", index=False)

# === Send Email ===
def send_email(to, subject, body, attachment, filename):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to
    msg.set_content(body)
    msg.add_attachment(attachment, maintype="application", subtype="octet-stream", filename=filename)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)

# === Streamlit App ===
st.set_page_config(page_title="Smart Investment App", layout="centered")
st.title("💹 Smart Compound + Recovery + Sharing")
tabs = st.tabs([
    "📈 Compound Calculator",
    "💸 Withdrawal Simulator",
    "🤝 Shared Investment Plan",
    "Range Summary Analysis",
    "📊 Jemey Real-Time Dashboard"
])


# === CONFIG ===
FULL_JEMEY_PASSWORD = "ahmedelite"  # Set your unlock password
LOGO_PATH = "C:/Users/ahmed/Downloads/jemeyai/logos/jemey.png"  # Your logo path
MODE_FILE = "jemey_mode.txt"  # For saving mode across sessions

# === Load saved mode if exists ===
if "jemey_mode" not in st.session_state:
    if os.path.exists(MODE_FILE):
        with open(MODE_FILE, "r") as f:
            st.session_state.jemey_mode = f.read().strip()
    else:
        st.session_state.jemey_mode = "Normal"

# === Convert image to base64 ===
def get_base64_image(LOGO_PATH):
    with open(LOGO_PATH, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# === Render Jemey Sidebar Logo & Title ===
def render_logo_sidebar():
    img_data = get_base64_image(LOGO_PATH)
    mode = st.session_state.get("jemey_mode", "Normal")

    if mode == "Full":
        jemey_title = "💠 JEMEY Engine"
        jemey_color = "#00f0ff"
    else:
        jemey_title = "🤖 JEMEY Normal"
        jemey_color = "#888888"

    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_data}" width="150" style="margin-bottom: 10px;">
            <h3 style="color:{jemey_color}; margin-top: 5px;">{jemey_title}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# === Unlock logic ===
if st.session_state.jemey_mode != "Full":
    st.sidebar.markdown("### 🔐 Jemey Access")
    auth_input = st.sidebar.text_input("Enter Unlock Key", type="password")
    if st.sidebar.button("🔓 Unlock Full Jemey"):
        if auth_input == FULL_JEMEY_PASSWORD:
            st.session_state.jemey_mode = "Full"
            with open(MODE_FILE, "w") as f:
                f.write("Full")
            st.sidebar.success("✅ Jemey Engine Activated")
            st.rerun()

        else:
            st.sidebar.error("❌ Wrong password. Try again.")
else:
    # Show "Lock" button if already in Full mode
    if st.sidebar.button("🔒 Lock Jemey (Back to Normal)"):
        st.session_state.jemey_mode = "Normal"
        with open(MODE_FILE, "w") as f:
            f.write("Normal")
        st.sidebar.info("🔐 Jemey is now in Normal Mode")
        st.rerun()


# === Show logo after logic ===
render_logo_sidebar()



# === Tab 1 ===
with tabs[0]:
    st.subheader("📈 Compound Interest Calculator")
    col1, col2 = st.columns(2)
    with col1:
        principal = st.number_input("Initial Investment", 500.0, step=50.0)
        rate = st.number_input("Interest Rate (%)", 2.217 , step=0.1)
        freq = st.selectbox("Compounding Frequency", ["Daily", "Weekly", "Monthly"])
    with col2:
        days = st.number_input("Days", 5, step=5)
        withdraw = st.checkbox("Withdraw profits?")
        currency = st.text_input("Currency Symbol", "$")

    auto_save = st.checkbox("Auto-save to history.csv")
    email = st.text_input("📧 Send result to email")

    if st.button("Calculate Compound Growth"):
        df, withdrawn, final = compound_growth(principal, rate, days, freq, withdraw)
        total_profit = df["Profit"].sum()

        st.success(f"💰 Total Profit: {currency}{total_profit:,.2f}")
        st.success(f"🏦 Final Balance: {currency}{final:,.2f}")
        if withdraw:
            st.info(f"💸 Withdrawn: {currency}{withdrawn:,.2f}")

        st.line_chart(df.set_index("Day")[["Balance", "Withdrawn"]])
        st.dataframe(df)
        excel = convert_df_to_excel(df)
        st.download_button("⬇️ Excel", data=excel, file_name="compound.xlsx")

        if auto_save:
            save_to_history({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Initial": principal,
                "Rate %": rate,
                "Days": days,
                "Profit": total_profit,
                "Final": final,
                "Withdrawn": withdrawn,
                "Mode": withdraw,
                "Frequency": freq
            })

        if email:
            try:
                send_email(email, "Compound Report", f"Final: {final:.2f}, Profit: {total_profit:.2f}", excel, "compound.xlsx")
                st.success("📧 Email sent!")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")

# === Tab 2 ===

with tabs[1]:
    st.subheader("💸 Daily Withdrawal Simulator")
    col1, col2 = st.columns(2)
    with col1:
        w_init = st.number_input("Initial Investment", 500.0, step=50.0, key="w1")
        w_days = st.number_input("Simulate for Days", 5, key="w2")
    with col2:
        w_rate = st.number_input("Daily Rate (%)", 2.217 , key="w3")
        fallback_ratio = st.number_input("Fallback Recovery Ratio (e.g. 0.67)", value=0.67, min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Simulate Withdrawal"):
        w_df, w_total, w_balance, status = profit_only_withdrawal(w_init, w_rate, w_days, fallback_ratio)
        st.info(status)
        st.success(f"💰 Withdrawn: {w_total:,.2f}")
        st.success(f"🏦 Final Balance: {w_balance:,.2f}")
        st.line_chart(w_df.set_index("Day")[["Balance", "Withdrawn"]])
        st.dataframe(w_df)


# === Tab 3 ===

with tabs[2]:
    st.subheader("🤝 Shared Strategy: Custom Profit Split & Fast Recovery")

    col1, col2 = st.columns(2)
    with col1:
        invest = st.number_input("Initial Investment", value=500.0, step=50.0)
        schedule_str = st.text_area("Gain Schedule (e.g. 1-10:3.3,11-20:2.2)", "1-10:3.3,11-20:2.2,21-30:4.4")
    with col2:
        split_ratio = st.number_input("Recovery Split Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        people_count = st.number_input("Number of People Sharing Profit", min_value=1, value=2, step=1)

    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    names, ratios = [], []
    st.markdown("### 🔧 Custom Profit Split Ratio")
    for i in range(1, people_count + 1):
        col1, col2 = st.columns([2, 1])
        with col1:
            name = st.text_input(f"Name of Person {i}", value=f"Person {i}", key=f"name{i}")
        with col2:
            ratio = st.number_input(f"{name}'s Share (%)", min_value=0.0, max_value=100.0,
                                    value=round(100 / people_count, 2), key=f"ratio{i}")
        names.append(name)
        ratios.append(ratio)

    total_ratio = sum(ratios)
    if abs(total_ratio - 100.0) > 0.01:
        st.error(f"⚠️ Total ratio must equal 100%. Current total: {total_ratio:.2f}%")
    else:
        if st.button("▶️ Run Custom Sharing Strategy"):
            rows = []
            balance = invest
            recovered = 0
            goal = invest
            day = 1
            max_days = 1000

            while recovered < goal and day <= max_days:
                rate = next((s['rate'] for s in schedule if s['from'] <= day <= s['to']), schedule[-1]['rate'])
                profit = balance * (rate / 100)
                recovery_part = profit * split_ratio
                shared_part = profit * (1 - split_ratio)

                recovered_today = min(recovery_part, goal - recovered)
                recovered += recovered_today
                balance += profit - recovered_today

                row = {
                    "Day": day,
                    "Rate %": rate,
                    "Profit": round(profit, 2),
                    "Recovered": round(recovered_today, 2),
                    "Balance": round(balance, 2),
                    "Remaining to Recover": round(goal - recovered, 2)
                }

                for name, ratio in zip(names, ratios):
                    row[f"{name} Share"] = round(shared_part * (ratio / 100), 2)

                rows.append(row)
                day += 1

            df = pd.DataFrame(rows)
            share_cols = [f"{name} Share" for name in names]

            st.session_state["df"] = df
            st.session_state["share_cols"] = share_cols

            if df.empty:
                st.error("⚠️ Recovery failed: no data generated.")
            else:
                st.success(f"✅ Full Recovery Achieved in {day - 1} days")
                st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
                st.dataframe(df)

              
# === Tab 4 ===

with tabs[3]:
    st.subheader("📊 Range Summary Analysis")

    if "df" in st.session_state and not st.session_state["df"].empty:
        df = st.session_state["df"]
        share_cols = st.session_state["share_cols"]

        max_day = int(df["Day"].max())
        from_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="range_from_day")
        to_day = st.number_input("End Day", min_value=from_day, max_value=max_day, value=from_day + 5, key="range_to_day")

        selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="range_cols")

        if st.button("✅ Analyze Range Summary"):
            try:
                range_df = df[(df["Day"] >= from_day) & (df["Day"] <= to_day)]

                if range_df.empty:
                    st.warning("⚠️ No data in selected range.")
                elif not selected_cols:
                    st.warning("⚠️ Please select people to analyze.")
                else:
                    selected_range = range_df[selected_cols]
                    st.subheader("📊 Totals")
                    st.dataframe(selected_range.sum().round(2).to_frame().T)

                    st.subheader("📈 Averages")
                    st.dataframe(selected_range.mean().round(2).to_frame().T)

                    st.subheader("🔺 Max")
                    st.dataframe(selected_range.max().round(2).to_frame().T)

                    st.subheader("🔻 Min")
                    st.dataframe(selected_range.min().round(2).to_frame().T)

                    st.line_chart(selected_range)

            except Exception as e:
                st.error(f"💥 Error during analysis: {e}")
    else:
        st.info("ℹ️ Run the sharing strategy in Tab 3 first to generate data.")


# === Tab 5 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)

with tabs[4]:
    st.title("📈 Jemey Live Market Dashboard")

    st.header("📊 Market Settings")
    asset_type = st.selectbox("Select Asset Type", ["Crypto", "Stock"])

    default_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"] if asset_type == "Crypto" else ["AAPL", "TSLA", "LE"]
    symbol = st.selectbox("Select Symbol", default_symbols)
    custom_symbol = st.text_input("Or enter custom symbol", "")
    active_symbol = custom_symbol if custom_symbol else symbol

    lookback = st.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

    @st.cache_data
    def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
        df = yf.download(tickers=symbol, period=period, interval=interval)
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        rename_map = {col: col.replace(f" {symbol}", "") for col in df.columns if f" {symbol}" in col}
        df.rename(columns=rename_map, inplace=True)
        return df

    try:
        df = fetch_market_data(active_symbol, period=lookback)

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            df.rename(columns={
                'Datetime': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
            }, inplace=True)

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in market data.")

        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
         
         # 🧮 Live Indicator Values
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        close = df['Close'].iloc[-1]
        bb_pos = (close - df['Low'].min()) / (df['High'].max() - df['Low'].min()) * 100
        atr_pct = (atr / close) * 100


        # Sample values to simulate the layout
        rsi = 54.58
        macd = -1036.76
        bb_position = "68.9%"
        volatility = "4.99%"

        ema20_50 = "Bearish"
        ema50_100 = "Bearish"
        ema100_200 = "Bullish"
        overall_trend = "Bearish"

        # Display section header
        st.markdown("## 📊 Technical Analysis Summary of `BTC-USD`")

        # Row 1: RSI, MACD, BB Position, Volatility (4 cards side by side)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### RSI")
            st.markdown("**Value:** 54.58")
            st.markdown("Normal Trading Range (RSI 30-70)")
            st.markdown("- Normal market conditions with balanced buying/selling")
            st.markdown("- Focus on trend following strategies")
            st.markdown("- Look for RSI direction and momentum")
            st.markdown("- Use other indicators for trade signals")

        with col2:
            st.markdown("### MACD")
            st.markdown("**Value:** -1036.76")
            st.markdown("Bullish MACD Cross")
            st.markdown("- Momentum is shifting bullish")
            st.markdown("- Consider long positions with positive histogram")
            st.markdown("- Stronger signal if crossover occurs below zero")
            st.markdown("- Use RSI and BB confirmation for better entries")

        with col3:
            st.markdown("### BB Position")
            st.markdown("**Value:** 68.9%")
            st.markdown("Price Within Bands")
            st.markdown("- Normal price movement within statistical range")
            st.markdown("- Watch for potential breakout if bands narrow")
            st.markdown("- Use other indicators to determine direction")
            st.markdown("- Consider setting alerts for band breaks")

        with col4:
            st.markdown("### Volatility (ATR)")
            st.markdown("**Value:** 4.99%")
            st.markdown("High Volatility (>3%)")
            st.markdown("- Large price movements are common")
            st.markdown("- Widen stop losses to account for swings")
            st.markdown("- Reduce position sizes to manage risk")
            st.markdown("- Consider staying out if too volatile")

        # Display section header
        st.markdown("## 📈 Moving Average Analysis")

        # Row 2: EMA Crosses
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.markdown("### EMA20/50 Cross")
            st.markdown("**Trend:** Bearish")
            st.markdown("Death Cross (Bearish)")
            st.markdown("- Shorter EMA crossed below longer EMA")
            st.markdown("- Strong bearish momentum signal")
            st.markdown("- Consider short positions with confirmation")
            st.markdown("- Use resistance levels for stop placement")

        with col6:
            st.markdown("### EMA50/100 Cross")
            st.markdown("**Trend:** Bearish")
            st.markdown("Death Cross (Bearish)")
            st.markdown("- Shorter EMA crossed below longer EMA")
            st.markdown("- Strong bearish momentum signal")
            st.markdown("- Consider short positions with confirmation")
            st.markdown("- Use resistance levels for stop placement")

        with col7:
            st.markdown("### EMA100/200 Cross")
            st.markdown("**Trend:** Bullish")
            st.markdown("Golden Cross (Bullish)")
            st.markdown("- Shorter EMA crossed above longer EMA")
            st.markdown("- Strong bullish momentum signal")
            st.markdown("- Consider long positions with confirmation")
            st.markdown("- Use support levels for stop placement")

        with col8:
            st.markdown("### Overall EMA Trend")
            st.markdown("**Trend:** Bearish")
            st.markdown("Strong Bearish Trend")
            st.markdown("- Multiple EMA crosses confirm downtrend")
            st.markdown("- Higher timeframe momentum is negative")
            st.markdown("- Look for rallies to EMAs as resistance")
            st.markdown("- Consider longer-term short positions")

        # 📊 Price + EMA Chart
        st.subheader(f"📈 Price Chart + EMA for {active_symbol}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Candles'))
        for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 📉 RSI
        st.subheader("📉 RSI")
        fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # 📈 MACD
        st.subheader("📈 MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
        fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
        st.plotly_chart(fig_macd, use_container_width=True)

        # 📊 ATR
        st.subheader("📊 ATR - Volatility")
        fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
        st.plotly_chart(fig_atr, use_container_width=True)

        # 💡 Risk Guidelines
        with st.expander("💡 Risk Management Guidelines"):
            st.markdown(""""
         **🔢 Position Sizing**
         - Risk 1–2% per trade
         - Adjust for volatility

            **🛑 Stop Loss Strategy**
            - Place SL outside noise
            - Use ATR or support/resistance

            **📉 Market Correlation**
            - Reduce exposure during high correlation
            - Diversify assets

            **🧠 Trade Management**
            - Use risk/reward of 1:2 or better
            - Never average down losers
            """)

        st.subheader("📂 Raw Market Data")
        st.dataframe(df)

        st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

    except Exception as e:
        st.error(f"⚠️ Failed to fetch market data: {e}")

    st.markdown("---")
    st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by JEMEY AI Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
# Smart Compound + Recovery + Sharing
import streamlit as st
from PIL import Image
import pandas as pd
from io import BytesIO
import smtplib
from email.message import EmailMessage
import os
import datetime
from PIL import Image
import re
import pytesseract
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import base64



# === Compound Growth ===
def compound_growth(principal, rate, days, frequency, withdraw):
    freq = {"Daily": 1, "Weekly": 7, "Monthly": 30}[frequency]
    data = []
    balance = principal
    withdrawn = 0

    for day in range(1, days + 1):
        if day % freq == 0:
            profit = balance * (rate / 100)
            if withdraw:
                withdrawn += profit
            else:
                balance += profit
        else:
            profit = 0

        data.append({
            "Day": day,
            "Profit": round(profit, 2),
            "Balance": round(balance, 2),
            "Withdrawn": round(withdrawn, 2)
        })

    return pd.DataFrame(data), withdrawn, balance

# === Profit-Only Withdrawal ===
def profit_only_withdrawal(initial, rate, days, fallback_ratio=2/3):
    balance = initial
    withdrawn = 0
    goal = initial * fallback_ratio
    data = []

    for day in range(1, days + 1):
        profit = balance * (rate / 100)
        take = min(profit, initial - withdrawn)
        balance += profit - take
        withdrawn += take

        data.append({
            "Day": day,
            "Daily Profit": round(profit, 2),
            "Withdrawn": round(take, 2),
            "Remaining": round(max(0, initial - withdrawn), 2),
            "Balance": round(balance, 2)
        })

        if withdrawn >= initial:
            break

    status = (
        f"✅ Full Recovery in {day} days"
        if withdrawn >= initial else
        f"⚠️ Partial Recovery: {withdrawn:.2f} ({(withdrawn / initial) * 100:.1f}%)"
    )

    return pd.DataFrame(data), withdrawn, balance, status

# === Shared Strategy ===
def shared_strategy(initial, schedule, split, people):
    balance = initial
    recovered = 0
    day = 1
    goal = initial
    rows = []

    while recovered < goal:
        rate = next((s['rate'] for s in schedule if s['from'] <= day <= s['to']), schedule[-1]['rate'])
        profit = balance * (rate / 100)
        rec_part = profit * split
        shared = profit * (1 - split)
        rec_now = min(rec_part, goal - recovered)
        recovered += rec_now
        balance += profit - rec_now

        row = {
            "Day": day,
            "Rate %": rate,
            "Profit": round(profit, 2),
            "Recovered": round(rec_now, 2),
            "Remaining to Recover": round(goal - recovered, 2),
            "Balance": round(balance, 2)
        }
        for i in range(1, people + 1):
            row[f"Person {i} Share"] = round(shared / people, 2)

        rows.append(row)
        day += 1

    return pd.DataFrame(rows), recovered, day

# === Export Excel ===
def convert_df_to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return out.getvalue()

# === Save History ===
def save_to_history(entry):
    df = pd.DataFrame([entry])
    if os.path.exists("history.csv"):
        df.to_csv("history.csv", mode="a", index=False, header=False)
    else:
        df.to_csv("history.csv", index=False)

# === Send Email ===
def send_email(to, subject, body, attachment, filename):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to
    msg.set_content(body)
    msg.add_attachment(attachment, maintype="application", subtype="octet-stream", filename=filename)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)

# === Streamlit App ===
st.set_page_config(page_title="Smart Investment App", layout="centered")
st.title("💹 Smart Compound + Recovery + Sharing")
tabs = st.tabs([
    "📈 Compound Calculator",
    "💸 Withdrawal Simulator",
    "🤝 Shared Investment Plan",
    "Range Summary Analysis",
    "📊 Jemey Real-Time Dashboard"
])


# === CONFIG ===
FULL_JEMEY_PASSWORD = "ahmedelite"  # Set your unlock password
LOGO_PATH = "C:/Users/ahmed/Downloads/jemeyai/logos/jemey.png"  # Your logo path
MODE_FILE = "jemey_mode.txt"  # For saving mode across sessions

# === Load saved mode if exists ===
if "jemey_mode" not in st.session_state:
    if os.path.exists(MODE_FILE):
        with open(MODE_FILE, "r") as f:
            st.session_state.jemey_mode = f.read().strip()
    else:
        st.session_state.jemey_mode = "Normal"

# === Convert image to base64 ===
def get_base64_image(LOGO_PATH):
    with open(LOGO_PATH, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# === Render Jemey Sidebar Logo & Title ===
def render_logo_sidebar():
    img_data = get_base64_image(LOGO_PATH)
    mode = st.session_state.get("jemey_mode", "Normal")

    if mode == "Full":
        jemey_title = "💠 JEMEY Engine"
        jemey_color = "#00f0ff"
    else:
        jemey_title = "🤖 JEMEY Normal"
        jemey_color = "#888888"

    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_data}" width="150" style="margin-bottom: 10px;">
            <h3 style="color:{jemey_color}; margin-top: 5px;">{jemey_title}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# === Unlock logic ===
if st.session_state.jemey_mode != "Full":
    st.sidebar.markdown("### 🔐 Jemey Access")
    auth_input = st.sidebar.text_input("Enter Unlock Key", type="password")
    if st.sidebar.button("🔓 Unlock Full Jemey"):
        if auth_input == FULL_JEMEY_PASSWORD:
            st.session_state.jemey_mode = "Full"
            with open(MODE_FILE, "w") as f:
                f.write("Full")
            st.sidebar.success("✅ Jemey Engine Activated")
            st.rerun()

        else:
            st.sidebar.error("❌ Wrong password. Try again.")
else:
    # Show "Lock" button if already in Full mode
    if st.sidebar.button("🔒 Lock Jemey (Back to Normal)"):
        st.session_state.jemey_mode = "Normal"
        with open(MODE_FILE, "w") as f:
            f.write("Normal")
        st.sidebar.info("🔐 Jemey is now in Normal Mode")
        st.rerun()


# === Show logo after logic ===
render_logo_sidebar()



# === Tab 1 ===
with tabs[0]:
    st.subheader("📈 Compound Interest Calculator")
    col1, col2 = st.columns(2)
    with col1:
        principal = st.number_input("Initial Investment", 500.0, step=50.0)
        rate = st.number_input("Interest Rate (%)", 2.217 , step=0.1)
        freq = st.selectbox("Compounding Frequency", ["Daily", "Weekly", "Monthly"])
    with col2:
        days = st.number_input("Days", 5, step=5)
        withdraw = st.checkbox("Withdraw profits?")
        currency = st.text_input("Currency Symbol", "$")

    auto_save = st.checkbox("Auto-save to history.csv")
    email = st.text_input("📧 Send result to email")

    if st.button("Calculate Compound Growth"):
        df, withdrawn, final = compound_growth(principal, rate, days, freq, withdraw)
        total_profit = df["Profit"].sum()

        st.success(f"💰 Total Profit: {currency}{total_profit:,.2f}")
        st.success(f"🏦 Final Balance: {currency}{final:,.2f}")
        if withdraw:
            st.info(f"💸 Withdrawn: {currency}{withdrawn:,.2f}")

        st.line_chart(df.set_index("Day")[["Balance", "Withdrawn"]])
        st.dataframe(df)
        excel = convert_df_to_excel(df)
        st.download_button("⬇️ Excel", data=excel, file_name="compound.xlsx")

        if auto_save:
            save_to_history({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Initial": principal,
                "Rate %": rate,
                "Days": days,
                "Profit": total_profit,
                "Final": final,
                "Withdrawn": withdrawn,
                "Mode": withdraw,
                "Frequency": freq
            })

        if email:
            try:
                send_email(email, "Compound Report", f"Final: {final:.2f}, Profit: {total_profit:.2f}", excel, "compound.xlsx")
                st.success("📧 Email sent!")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")

# === Tab 2 ===

with tabs[1]:
    st.subheader("💸 Daily Withdrawal Simulator")
    col1, col2 = st.columns(2)
    with col1:
        w_init = st.number_input("Initial Investment", 500.0, step=50.0, key="w1")
        w_days = st.number_input("Simulate for Days", 5, key="w2")
    with col2:
        w_rate = st.number_input("Daily Rate (%)", 2.217 , key="w3")
        fallback_ratio = st.number_input("Fallback Recovery Ratio (e.g. 0.67)", value=0.67, min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Simulate Withdrawal"):
        w_df, w_total, w_balance, status = profit_only_withdrawal(w_init, w_rate, w_days, fallback_ratio)
        st.info(status)
        st.success(f"💰 Withdrawn: {w_total:,.2f}")
        st.success(f"🏦 Final Balance: {w_balance:,.2f}")
        st.line_chart(w_df.set_index("Day")[["Balance", "Withdrawn"]])
        st.dataframe(w_df)


# === Tab 3 ===

with tabs[2]:
    st.subheader("🤝 Shared Strategy: Custom Profit Split & Fast Recovery")

    col1, col2 = st.columns(2)
    with col1:
        invest = st.number_input("Initial Investment", value=500.0, step=50.0)
        schedule_str = st.text_area("Gain Schedule (e.g. 1-10:3.3,11-20:2.2)", "1-10:3.3,11-20:2.2,21-30:4.4")
    with col2:
        split_ratio = st.number_input("Recovery Split Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        people_count = st.number_input("Number of People Sharing Profit", min_value=1, value=2, step=1)

    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    names, ratios = [], []
    st.markdown("### 🔧 Custom Profit Split Ratio")
    for i in range(1, people_count + 1):
        col1, col2 = st.columns([2, 1])
        with col1:
            name = st.text_input(f"Name of Person {i}", value=f"Person {i}", key=f"name{i}")
        with col2:
            ratio = st.number_input(f"{name}'s Share (%)", min_value=0.0, max_value=100.0,
                                    value=round(100 / people_count, 2), key=f"ratio{i}")
        names.append(name)
        ratios.append(ratio)

    total_ratio = sum(ratios)
    if abs(total_ratio - 100.0) > 0.01:
        st.error(f"⚠️ Total ratio must equal 100%. Current total: {total_ratio:.2f}%")
    else:
        if st.button("▶️ Run Custom Sharing Strategy"):
            rows = []
            balance = invest
            recovered = 0
            goal = invest
            day = 1
            max_days = 1000

            while recovered < goal and day <= max_days:
                rate = next((s['rate'] for s in schedule if s['from'] <= day <= s['to']), schedule[-1]['rate'])
                profit = balance * (rate / 100)
                recovery_part = profit * split_ratio
                shared_part = profit * (1 - split_ratio)

                recovered_today = min(recovery_part, goal - recovered)
                recovered += recovered_today
                balance += profit - recovered_today

                row = {
                    "Day": day,
                    "Rate %": rate,
                    "Profit": round(profit, 2),
                    "Recovered": round(recovered_today, 2),
                    "Balance": round(balance, 2),
                    "Remaining to Recover": round(goal - recovered, 2)
                }

                for name, ratio in zip(names, ratios):
                    row[f"{name} Share"] = round(shared_part * (ratio / 100), 2)

                rows.append(row)
                day += 1

            df = pd.DataFrame(rows)
            share_cols = [f"{name} Share" for name in names]

            st.session_state["df"] = df
            st.session_state["share_cols"] = share_cols

            if df.empty:
                st.error("⚠️ Recovery failed: no data generated.")
            else:
                st.success(f"✅ Full Recovery Achieved in {day - 1} days")
                st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
                st.dataframe(df)

              
# === Tab 4 ===

with tabs[3]:
    st.subheader("📊 Range Summary Analysis")

    if "df" in st.session_state and not st.session_state["df"].empty:
        df = st.session_state["df"]
        share_cols = st.session_state["share_cols"]

        max_day = int(df["Day"].max())
        from_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="range_from_day")
        to_day = st.number_input("End Day", min_value=from_day, max_value=max_day, value=from_day + 5, key="range_to_day")

        selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="range_cols")

        if st.button("✅ Analyze Range Summary"):
            try:
                range_df = df[(df["Day"] >= from_day) & (df["Day"] <= to_day)]

                if range_df.empty:
                    st.warning("⚠️ No data in selected range.")
                elif not selected_cols:
                    st.warning("⚠️ Please select people to analyze.")
                else:
                    selected_range = range_df[selected_cols]
                    st.subheader("📊 Totals")
                    st.dataframe(selected_range.sum().round(2).to_frame().T)

                    st.subheader("📈 Averages")
                    st.dataframe(selected_range.mean().round(2).to_frame().T)

                    st.subheader("🔺 Max")
                    st.dataframe(selected_range.max().round(2).to_frame().T)

                    st.subheader("🔻 Min")
                    st.dataframe(selected_range.min().round(2).to_frame().T)

                    st.line_chart(selected_range)

            except Exception as e:
                st.error(f"💥 Error during analysis: {e}")
    else:
        st.info("ℹ️ Run the sharing strategy in Tab 3 first to generate data.")


# === Tab 5 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)

with tabs[4]:
    st.title("📈 Jemey Live Market Dashboard")

    st.header("📊 Market Settings")
    asset_type = st.selectbox("Select Asset Type", ["Crypto", "Stock"])

    default_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"] if asset_type == "Crypto" else ["AAPL", "TSLA", "LE"]
    symbol = st.selectbox("Select Symbol", default_symbols)
    custom_symbol = st.text_input("Or enter custom symbol", "")
    active_symbol = custom_symbol if custom_symbol else symbol

    lookback = st.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

    @st.cache_data
    def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
        df = yf.download(tickers=symbol, period=period, interval=interval)
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        rename_map = {col: col.replace(f" {symbol}", "") for col in df.columns if f" {symbol}" in col}
        df.rename(columns=rename_map, inplace=True)
        return df

    try:
        df = fetch_market_data(active_symbol, period=lookback)

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            df.rename(columns={
                'Datetime': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
            }, inplace=True)

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in market data.")

        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
         
         # 🧮 Live Indicator Values
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        close = df['Close'].iloc[-1]
        bb_pos = (close - df['Low'].min()) / (df['High'].max() - df['Low'].min()) * 100
        atr_pct = (atr / close) * 100


        # Sample values to simulate the layout
        rsi = 54.58
        macd = -1036.76
        bb_position = "68.9%"
        volatility = "4.99%"

        ema20_50 = "Bearish"
        ema50_100 = "Bearish"
        ema100_200 = "Bullish"
        overall_trend = "Bearish"

        # Display section header
        st.markdown("## 📊 Technical Analysis Summary of `BTC-USD`")

        # Row 1: RSI, MACD, BB Position, Volatility (4 cards side by side)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### RSI")
            st.markdown("**Value:** 54.58")
            st.markdown("Normal Trading Range (RSI 30-70)")
            st.markdown("- Normal market conditions with balanced buying/selling")
            st.markdown("- Focus on trend following strategies")
            st.markdown("- Look for RSI direction and momentum")
            st.markdown("- Use other indicators for trade signals")

        with col2:
            st.markdown("### MACD")
            st.markdown("**Value:** -1036.76")
            st.markdown("Bullish MACD Cross")
            st.markdown("- Momentum is shifting bullish")
            st.markdown("- Consider long positions with positive histogram")
            st.markdown("- Stronger signal if crossover occurs below zero")
            st.markdown("- Use RSI and BB confirmation for better entries")

        with col3:
            st.markdown("### BB Position")
            st.markdown("**Value:** 68.9%")
            st.markdown("Price Within Bands")
            st.markdown("- Normal price movement within statistical range")
            st.markdown("- Watch for potential breakout if bands narrow")
            st.markdown("- Use other indicators to determine direction")
            st.markdown("- Consider setting alerts for band breaks")

        with col4:
            st.markdown("### Volatility (ATR)")
            st.markdown("**Value:** 4.99%")
            st.markdown("High Volatility (>3%)")
            st.markdown("- Large price movements are common")
            st.markdown("- Widen stop losses to account for swings")
            st.markdown("- Reduce position sizes to manage risk")
            st.markdown("- Consider staying out if too volatile")

        # Display section header
        st.markdown("## 📈 Moving Average Analysis")

        # Row 2: EMA Crosses
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.markdown("### EMA20/50 Cross")
            st.markdown("**Trend:** Bearish")
            st.markdown("Death Cross (Bearish)")
            st.markdown("- Shorter EMA crossed below longer EMA")
            st.markdown("- Strong bearish momentum signal")
            st.markdown("- Consider short positions with confirmation")
            st.markdown("- Use resistance levels for stop placement")

        with col6:
            st.markdown("### EMA50/100 Cross")
            st.markdown("**Trend:** Bearish")
            st.markdown("Death Cross (Bearish)")
            st.markdown("- Shorter EMA crossed below longer EMA")
            st.markdown("- Strong bearish momentum signal")
            st.markdown("- Consider short positions with confirmation")
            st.markdown("- Use resistance levels for stop placement")

        with col7:
            st.markdown("### EMA100/200 Cross")
            st.markdown("**Trend:** Bullish")
            st.markdown("Golden Cross (Bullish)")
            st.markdown("- Shorter EMA crossed above longer EMA")
            st.markdown("- Strong bullish momentum signal")
            st.markdown("- Consider long positions with confirmation")
            st.markdown("- Use support levels for stop placement")

        with col8:
            st.markdown("### Overall EMA Trend")
            st.markdown("**Trend:** Bearish")
            st.markdown("Strong Bearish Trend")
            st.markdown("- Multiple EMA crosses confirm downtrend")
            st.markdown("- Higher timeframe momentum is negative")
            st.markdown("- Look for rallies to EMAs as resistance")
            st.markdown("- Consider longer-term short positions")

        # 📊 Price + EMA Chart
        st.subheader(f"📈 Price Chart + EMA for {active_symbol}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Candles'))
        for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 📉 RSI
        st.subheader("📉 RSI")
        fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # 📈 MACD
        st.subheader("📈 MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
        fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
        st.plotly_chart(fig_macd, use_container_width=True)

        # 📊 ATR
        st.subheader("📊 ATR - Volatility")
        fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
        st.plotly_chart(fig_atr, use_container_width=True)

        # 💡 Risk Guidelines
        with st.expander("💡 Risk Management Guidelines"):
            st.markdown(""""
         **🔢 Position Sizing**
         - Risk 1–2% per trade
         - Adjust for volatility

            **🛑 Stop Loss Strategy**
            - Place SL outside noise
            - Use ATR or support/resistance

            **📉 Market Correlation**
            - Reduce exposure during high correlation
            - Diversify assets

            **🧠 Trade Management**
            - Use risk/reward of 1:2 or better
            - Never average down losers
            """)

        st.subheader("📂 Raw Market Data")
        st.dataframe(df)

        st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

    except Exception as e:
        st.error(f"⚠️ Failed to fetch market data: {e}")

    st.markdown("---")
    st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by JEMEY AI Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
# Smart Compound + Recovery + Sharing
import streamlit as st
from PIL import Image
import pandas as pd
from io import BytesIO
import smtplib
from email.message import EmailMessage
import os
import datetime
from PIL import Image
import re
import pytesseract
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import base64



# === Compound Growth ===
def compound_growth(principal, rate, days, frequency, withdraw):
    freq = {"Daily": 1, "Weekly": 7, "Monthly": 30}[frequency]
    data = []
    balance = principal
    withdrawn = 0

    for day in range(1, days + 1):
        if day % freq == 0:
            profit = balance * (rate / 100)
            if withdraw:
                withdrawn += profit
            else:
                balance += profit
        else:
            profit = 0

        data.append({
            "Day": day,
            "Profit": round(profit, 2),
            "Balance": round(balance, 2),
            "Withdrawn": round(withdrawn, 2)
        })

    return pd.DataFrame(data), withdrawn, balance

# === Profit-Only Withdrawal ===
def profit_only_withdrawal(initial, rate, days, fallback_ratio=2/3):
    balance = initial
    withdrawn = 0
    goal = initial * fallback_ratio
    data = []

    for day in range(1, days + 1):
        profit = balance * (rate / 100)
        take = min(profit, initial - withdrawn)
        balance += profit - take
        withdrawn += take

        data.append({
            "Day": day,
            "Daily Profit": round(profit, 2),
            "Withdrawn": round(take, 2),
            "Remaining": round(max(0, initial - withdrawn), 2),
            "Balance": round(balance, 2)
        })

        if withdrawn >= initial:
            break

    status = (
        f"✅ Full Recovery in {day} days"
        if withdrawn >= initial else
        f"⚠️ Partial Recovery: {withdrawn:.2f} ({(withdrawn / initial) * 100:.1f}%)"
    )

    return pd.DataFrame(data), withdrawn, balance, status

# === Shared Strategy ===
def shared_strategy(initial, schedule, split, people):
    balance = initial
    recovered = 0
    day = 1
    goal = initial
    rows = []

    while recovered < goal:
        rate = next((s['rate'] for s in schedule if s['from'] <= day <= s['to']), schedule[-1]['rate'])
        profit = balance * (rate / 100)
        rec_part = profit * split
        shared = profit * (1 - split)
        rec_now = min(rec_part, goal - recovered)
        recovered += rec_now
        balance += profit - rec_now

        row = {
            "Day": day,
            "Rate %": rate,
            "Profit": round(profit, 2),
            "Recovered": round(rec_now, 2),
            "Remaining to Recover": round(goal - recovered, 2),
            "Balance": round(balance, 2)
        }
        for i in range(1, people + 1):
            row[f"Person {i} Share"] = round(shared / people, 2)

        rows.append(row)
        day += 1

    return pd.DataFrame(rows), recovered, day

# === Export Excel ===
def convert_df_to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return out.getvalue()

# === Save History ===
def save_to_history(entry):
    df = pd.DataFrame([entry])
    if os.path.exists("history.csv"):
        df.to_csv("history.csv", mode="a", index=False, header=False)
    else:
        df.to_csv("history.csv", index=False)

# === Send Email ===
def send_email(to, subject, body, attachment, filename):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to
    msg.set_content(body)
    msg.add_attachment(attachment, maintype="application", subtype="octet-stream", filename=filename)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)

# === Streamlit App ===
st.set_page_config(page_title="Smart Investment App", layout="centered")
st.title("💹 Smart Compound + Recovery + Sharing")
tabs = st.tabs([
    "📈 Compound Calculator",
    "💸 Withdrawal Simulator",
    "🤝 Shared Investment Plan",
    "Range Summary Analysis",
    "📊 Jemey Real-Time Dashboard"
])


# === CONFIG ===
FULL_JEMEY_PASSWORD = "ahmedelite"  # Set your unlock password
LOGO_PATH = "C:/Users/ahmed/Downloads/jemeyai/logos/jemey.png"  # Your logo path
MODE_FILE = "jemey_mode.txt"  # For saving mode across sessions

# === Load saved mode if exists ===
if "jemey_mode" not in st.session_state:
    if os.path.exists(MODE_FILE):
        with open(MODE_FILE, "r") as f:
            st.session_state.jemey_mode = f.read().strip()
    else:
        st.session_state.jemey_mode = "Normal"

# === Convert image to base64 ===
def get_base64_image(LOGO_PATH):
    with open(LOGO_PATH, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# === Render Jemey Sidebar Logo & Title ===
def render_logo_sidebar():
    img_data = get_base64_image(LOGO_PATH)
    mode = st.session_state.get("jemey_mode", "Normal")

    if mode == "Full":
        jemey_title = "💠 JEMEY Engine"
        jemey_color = "#00f0ff"
    else:
        jemey_title = "🤖 JEMEY Normal"
        jemey_color = "#888888"

    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_data}" width="150" style="margin-bottom: 10px;">
            <h3 style="color:{jemey_color}; margin-top: 5px;">{jemey_title}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# === Unlock logic ===
if st.session_state.jemey_mode != "Full":
    st.sidebar.markdown("### 🔐 Jemey Access")
    auth_input = st.sidebar.text_input("Enter Unlock Key", type="password")
    if st.sidebar.button("🔓 Unlock Full Jemey"):
        if auth_input == FULL_JEMEY_PASSWORD:
            st.session_state.jemey_mode = "Full"
            with open(MODE_FILE, "w") as f:
                f.write("Full")
            st.sidebar.success("✅ Jemey Engine Activated")
            st.rerun()

        else:
            st.sidebar.error("❌ Wrong password. Try again.")
else:
    # Show "Lock" button if already in Full mode
    if st.sidebar.button("🔒 Lock Jemey (Back to Normal)"):
        st.session_state.jemey_mode = "Normal"
        with open(MODE_FILE, "w") as f:
            f.write("Normal")
        st.sidebar.info("🔐 Jemey is now in Normal Mode")
        st.rerun()


# === Show logo after logic ===
render_logo_sidebar()



# === Tab 1 ===
with tabs[0]:
    st.subheader("📈 Compound Interest Calculator")
    col1, col2 = st.columns(2)
    with col1:
        principal = st.number_input("Initial Investment", 500.0, step=50.0)
        rate = st.number_input("Interest Rate (%)", 2.217 , step=0.1)
        freq = st.selectbox("Compounding Frequency", ["Daily", "Weekly", "Monthly"])
    with col2:
        days = st.number_input("Days", 5, step=5)
        withdraw = st.checkbox("Withdraw profits?")
        currency = st.text_input("Currency Symbol", "$")

    auto_save = st.checkbox("Auto-save to history.csv")
    email = st.text_input("📧 Send result to email")

    if st.button("Calculate Compound Growth"):
        df, withdrawn, final = compound_growth(principal, rate, days, freq, withdraw)
        total_profit = df["Profit"].sum()

        st.success(f"💰 Total Profit: {currency}{total_profit:,.2f}")
        st.success(f"🏦 Final Balance: {currency}{final:,.2f}")
        if withdraw:
            st.info(f"💸 Withdrawn: {currency}{withdrawn:,.2f}")

        st.line_chart(df.set_index("Day")[["Balance", "Withdrawn"]])
        st.dataframe(df)
        excel = convert_df_to_excel(df)
        st.download_button("⬇️ Excel", data=excel, file_name="compound.xlsx")

        if auto_save:
            save_to_history({
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Initial": principal,
                "Rate %": rate,
                "Days": days,
                "Profit": total_profit,
                "Final": final,
                "Withdrawn": withdrawn,
                "Mode": withdraw,
                "Frequency": freq
            })

        if email:
            try:
                send_email(email, "Compound Report", f"Final: {final:.2f}, Profit: {total_profit:.2f}", excel, "compound.xlsx")
                st.success("📧 Email sent!")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")

# === Tab 2 ===

with tabs[1]:
    st.subheader("💸 Daily Withdrawal Simulator")
    col1, col2 = st.columns(2)
    with col1:
        w_init = st.number_input("Initial Investment", 500.0, step=50.0, key="w1")
        w_days = st.number_input("Simulate for Days", 5, key="w2")
    with col2:
        w_rate = st.number_input("Daily Rate (%)", 2.217 , key="w3")
        fallback_ratio = st.number_input("Fallback Recovery Ratio (e.g. 0.67)", value=0.67, min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Simulate Withdrawal"):
        w_df, w_total, w_balance, status = profit_only_withdrawal(w_init, w_rate, w_days, fallback_ratio)
        st.info(status)
        st.success(f"💰 Withdrawn: {w_total:,.2f}")
        st.success(f"🏦 Final Balance: {w_balance:,.2f}")
        st.line_chart(w_df.set_index("Day")[["Balance", "Withdrawn"]])
        st.dataframe(w_df)


# === Tab 3 ===

with tabs[2]:
    st.subheader("🤝 Shared Strategy: Custom Profit Split & Fast Recovery")

    col1, col2 = st.columns(2)
    with col1:
        invest = st.number_input("Initial Investment", value=500.0, step=50.0)
        schedule_str = st.text_area("Gain Schedule (e.g. 1-10:3.3,11-20:2.2)", "1-10:3.3,11-20:2.2,21-30:4.4")
    with col2:
        split_ratio = st.number_input("Recovery Split Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        people_count = st.number_input("Number of People Sharing Profit", min_value=1, value=2, step=1)

    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    names, ratios = [], []
    st.markdown("### 🔧 Custom Profit Split Ratio")
    for i in range(1, people_count + 1):
        col1, col2 = st.columns([2, 1])
        with col1:
            name = st.text_input(f"Name of Person {i}", value=f"Person {i}", key=f"name{i}")
        with col2:
            ratio = st.number_input(f"{name}'s Share (%)", min_value=0.0, max_value=100.0,
                                    value=round(100 / people_count, 2), key=f"ratio{i}")
        names.append(name)
        ratios.append(ratio)

    total_ratio = sum(ratios)
    if abs(total_ratio - 100.0) > 0.01:
        st.error(f"⚠️ Total ratio must equal 100%. Current total: {total_ratio:.2f}%")
    else:
        if st.button("▶️ Run Custom Sharing Strategy"):
            rows = []
            balance = invest
            recovered = 0
            goal = invest
            day = 1
            max_days = 1000

            while recovered < goal and day <= max_days:
                rate = next((s['rate'] for s in schedule if s['from'] <= day <= s['to']), schedule[-1]['rate'])
                profit = balance * (rate / 100)
                recovery_part = profit * split_ratio
                shared_part = profit * (1 - split_ratio)

                recovered_today = min(recovery_part, goal - recovered)
                recovered += recovered_today
                balance += profit - recovered_today

                row = {
                    "Day": day,
                    "Rate %": rate,
                    "Profit": round(profit, 2),
                    "Recovered": round(recovered_today, 2),
                    "Balance": round(balance, 2),
                    "Remaining to Recover": round(goal - recovered, 2)
                }

                for name, ratio in zip(names, ratios):
                    row[f"{name} Share"] = round(shared_part * (ratio / 100), 2)

                rows.append(row)
                day += 1

            df = pd.DataFrame(rows)
            share_cols = [f"{name} Share" for name in names]

            st.session_state["df"] = df
            st.session_state["share_cols"] = share_cols

            if df.empty:
                st.error("⚠️ Recovery failed: no data generated.")
            else:
                st.success(f"✅ Full Recovery Achieved in {day - 1} days")
                st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
                st.dataframe(df)

              
# === Tab 4 ===

with tabs[3]:
    st.subheader("📊 Range Summary Analysis")

    if "df" in st.session_state and not st.session_state["df"].empty:
        df = st.session_state["df"]
        share_cols = st.session_state["share_cols"]

        max_day = int(df["Day"].max())
        from_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="range_from_day")
        to_day = st.number_input("End Day", min_value=from_day, max_value=max_day, value=from_day + 5, key="range_to_day")

        selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="range_cols")

        if st.button("✅ Analyze Range Summary"):
            try:
                range_df = df[(df["Day"] >= from_day) & (df["Day"] <= to_day)]

                if range_df.empty:
                    st.warning("⚠️ No data in selected range.")
                elif not selected_cols:
                    st.warning("⚠️ Please select people to analyze.")
                else:
                    selected_range = range_df[selected_cols]
                    st.subheader("📊 Totals")
                    st.dataframe(selected_range.sum().round(2).to_frame().T)

                    st.subheader("📈 Averages")
                    st.dataframe(selected_range.mean().round(2).to_frame().T)

                    st.subheader("🔺 Max")
                    st.dataframe(selected_range.max().round(2).to_frame().T)

                    st.subheader("🔻 Min")
                    st.dataframe(selected_range.min().round(2).to_frame().T)

                    st.line_chart(selected_range)

            except Exception as e:
                st.error(f"💥 Error during analysis: {e}")
    else:
        st.info("ℹ️ Run the sharing strategy in Tab 3 first to generate data.")


# === Tab 5 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)

with tabs[4]:
    st.title("📈 Jemey Live Market Dashboard")

    st.header("📊 Market Settings")
    asset_type = st.selectbox("Select Asset Type", ["Crypto", "Stock"])

    default_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"] if asset_type == "Crypto" else ["AAPL", "TSLA", "LE"]
    symbol = st.selectbox("Select Symbol", default_symbols)
    custom_symbol = st.text_input("Or enter custom symbol", "")
    active_symbol = custom_symbol if custom_symbol else symbol

    lookback = st.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

    @st.cache_data
    def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
        df = yf.download(tickers=symbol, period=period, interval=interval)
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        rename_map = {col: col.replace(f" {symbol}", "") for col in df.columns if f" {symbol}" in col}
        df.rename(columns=rename_map, inplace=True)
        return df

    try:
        df = fetch_market_data(active_symbol, period=lookback)

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            df.rename(columns={
                'Datetime': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
            }, inplace=True)

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in market data.")

        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
         
         # 🧮 Live Indicator Values
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        signal = df['Signal'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        close = df['Close'].iloc[-1]
        bb_pos = (close - df['Low'].min()) / (df['High'].max() - df['Low'].min()) * 100
        atr_pct = (atr / close) * 100


        # Sample values to simulate the layout
        rsi = 54.58
        macd = -1036.76
        bb_position = "68.9%"
        volatility = "4.99%"

        ema20_50 = "Bearish"
        ema50_100 = "Bearish"
        ema100_200 = "Bullish"
        overall_trend = "Bearish"

        # Display section header
        st.markdown("## 📊 Technical Analysis Summary of `BTC-USD`")

        # Row 1: RSI, MACD, BB Position, Volatility (4 cards side by side)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### RSI")
            st.markdown("**Value:** 54.58")
            st.markdown("Normal Trading Range (RSI 30-70)")
            st.markdown("- Normal market conditions with balanced buying/selling")
            st.markdown("- Focus on trend following strategies")
            st.markdown("- Look for RSI direction and momentum")
            st.markdown("- Use other indicators for trade signals")

        with col2:
            st.markdown("### MACD")
            st.markdown("**Value:** -1036.76")
            st.markdown("Bullish MACD Cross")
            st.markdown("- Momentum is shifting bullish")
            st.markdown("- Consider long positions with positive histogram")
            st.markdown("- Stronger signal if crossover occurs below zero")
            st.markdown("- Use RSI and BB confirmation for better entries")

        with col3:
            st.markdown("### BB Position")
            st.markdown("**Value:** 68.9%")
            st.markdown("Price Within Bands")
            st.markdown("- Normal price movement within statistical range")
            st.markdown("- Watch for potential breakout if bands narrow")
            st.markdown("- Use other indicators to determine direction")
            st.markdown("- Consider setting alerts for band breaks")

        with col4:
            st.markdown("### Volatility (ATR)")
            st.markdown("**Value:** 4.99%")
            st.markdown("High Volatility (>3%)")
            st.markdown("- Large price movements are common")
            st.markdown("- Widen stop losses to account for swings")
            st.markdown("- Reduce position sizes to manage risk")
            st.markdown("- Consider staying out if too volatile")

        # Display section header
        st.markdown("## 📈 Moving Average Analysis")

        # Row 2: EMA Crosses
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.markdown("### EMA20/50 Cross")
            st.markdown("**Trend:** Bearish")
            st.markdown("Death Cross (Bearish)")
            st.markdown("- Shorter EMA crossed below longer EMA")
            st.markdown("- Strong bearish momentum signal")
            st.markdown("- Consider short positions with confirmation")
            st.markdown("- Use resistance levels for stop placement")

        with col6:
            st.markdown("### EMA50/100 Cross")
            st.markdown("**Trend:** Bearish")
            st.markdown("Death Cross (Bearish)")
            st.markdown("- Shorter EMA crossed below longer EMA")
            st.markdown("- Strong bearish momentum signal")
            st.markdown("- Consider short positions with confirmation")
            st.markdown("- Use resistance levels for stop placement")

        with col7:
            st.markdown("### EMA100/200 Cross")
            st.markdown("**Trend:** Bullish")
            st.markdown("Golden Cross (Bullish)")
            st.markdown("- Shorter EMA crossed above longer EMA")
            st.markdown("- Strong bullish momentum signal")
            st.markdown("- Consider long positions with confirmation")
            st.markdown("- Use support levels for stop placement")

        with col8:
            st.markdown("### Overall EMA Trend")
            st.markdown("**Trend:** Bearish")
            st.markdown("Strong Bearish Trend")
            st.markdown("- Multiple EMA crosses confirm downtrend")
            st.markdown("- Higher timeframe momentum is negative")
            st.markdown("- Look for rallies to EMAs as resistance")
            st.markdown("- Consider longer-term short positions")

        # 📊 Price + EMA Chart
        st.subheader(f"📈 Price Chart + EMA for {active_symbol}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Candles'))
        for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 📉 RSI
        st.subheader("📉 RSI")
        fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # 📈 MACD
        st.subheader("📈 MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
        fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
        st.plotly_chart(fig_macd, use_container_width=True)

        # 📊 ATR
        st.subheader("📊 ATR - Volatility")
        fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
        st.plotly_chart(fig_atr, use_container_width=True)

        # 💡 Risk Guidelines
        with st.expander("💡 Risk Management Guidelines"):
            st.markdown(""""
         **🔢 Position Sizing**
         - Risk 1–2% per trade
         - Adjust for volatility

            **🛑 Stop Loss Strategy**
            - Place SL outside noise
            - Use ATR or support/resistance

            **📉 Market Correlation**
            - Reduce exposure during high correlation
            - Diversify assets

            **🧠 Trade Management**
            - Use risk/reward of 1:2 or better
            - Never average down losers
            """)

        st.subheader("📂 Raw Market Data")
        st.dataframe(df)

        st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

    except Exception as e:
        st.error(f"⚠️ Failed to fetch market data: {e}")

    st.markdown("---")
    st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by JEMEY AI Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
