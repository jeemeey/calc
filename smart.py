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
    "📊 Jemey Real-Time Dashboard"
])

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

    # Parse schedule
    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    # Custom names and ratios
    names = []
    ratios = []
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

            while recovered < goal:
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
            st.success(f"✅ Full Recovery Achieved in {day - 1} days")

            share_cols = [f"{name} Share" for name in names]
            st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
            st.dataframe(df)

            # Manual day range inputs
            st.markdown("### 🎯 Analyze by Manual Day Range")
            max_day = int(df["Day"].max())
            start_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="start_day_input")
            end_day = st.number_input("End Day", min_value=start_day, max_value=max_day, value=max_day, key="end_day_input")

            range_df = df[(df["Day"] >= start_day) & (df["Day"] <= end_day)]
            selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="manual_range_cols")

            if selected_cols:
             try:
                 selected_range = range_df[selected_cols]

                 if selected_range.empty:
                  st.warning("⚠️ No data in selected range.")
                 else:
                   st.write("📊 Totals:", selected_range.sum().round(2))
                   st.write("📈 Averages:", selected_range.mean().round(2))
                   st.write("🔺 Max:", selected_range.max().round(2))
                   st.write("🔻 Min:", selected_range.min().round(2))
                   st.line_chart(selected_range)
             except KeyError as e:
                   st.error(f"🚫 Invalid column selection: {e}")
             except Exception as e:
                   st.error(f"💥 Unexpected error: {e}")
            else:
                   st.info("Choose at least one column to view stats.")


# === Tab 4 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)
with tabs[3]:
   
 st.title("📈 Jemey Live Market Dashboard")

 st.sidebar.header("📊 Market Settings")
 asset_type = st.sidebar.radio("Select Asset Type", ["Crypto", "Stock"])
 symbol = st.sidebar.selectbox("Choose Symbol", ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"])
 lookback = st.sidebar.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

 @st.cache_data
 def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.reset_index(inplace=True)

    # Flatten column names if needed (only if multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Rename columns if they have ticker prefix
    rename_map = {}
    for col in df.columns:
        if f" {symbol}" in col:
            rename_map[col] = col.replace(f" {symbol}", "")
    df.rename(columns=rename_map, inplace=True)

    return df

 try:
     df = fetch_market_data(symbol, period=lookback)

     # Check required columns
     required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
     if not all(col in df.columns for col in required_columns):
      st.warning("Trying alternative fallback column structure...")
     df.rename(columns={
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
        }, inplace=True)

     # Validate again
     if not all(col in df.columns for col in required_columns):
      raise ValueError("One or more required columns are missing in the data.")

     df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
     df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
     df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
     df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
     df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
     df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
     df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
     df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
     df['MACD_Hist'] = df['MACD'] - df['Signal']

     st.subheader(f"📊 Chart for {symbol} ({lookback})")

     fig = go.Figure()
     fig.add_trace(go.Candlestick(
       x=df['Date'],
       open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
       name='Candles'))
     for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
      fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))

     fig.update_layout(title='Price Chart + EMA', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
     st.plotly_chart(fig, use_container_width=True)

     st.markdown("---")
     st.subheader("📉 RSI")
     fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
     fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
     fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
     st.plotly_chart(fig_rsi, use_container_width=True)

     st.subheader("📈 MACD")
     fig_macd = go.Figure()
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
     fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
     st.plotly_chart(fig_macd, use_container_width=True)

     st.subheader("📊 ATR - Volatility")
     fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
     st.plotly_chart(fig_atr, use_container_width=True)

     with st.expander("💡 Risk Management Guidelines"):
         st.markdown("""
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

     st.markdown("---")
     st.subheader("📂 Raw Market Data")
     st.dataframe(df)

     st.markdown("---")
     st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

 except Exception as e:
     st.error(f"⚠️ Failed to fetch market data: {e}")


 st.markdown("---")
 st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
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
    "📊 Jemey Real-Time Dashboard"
])

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

    # Parse schedule
    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    # Custom names and ratios
    names = []
    ratios = []
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

            while recovered < goal:
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
            st.success(f"✅ Full Recovery Achieved in {day - 1} days")

            share_cols = [f"{name} Share" for name in names]
            st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
            st.dataframe(df)

            # Manual day range inputs
            st.markdown("### 🎯 Analyze by Manual Day Range")
            max_day = int(df["Day"].max())
            start_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="start_day_input")
            end_day = st.number_input("End Day", min_value=start_day, max_value=max_day, value=max_day, key="end_day_input")

            range_df = df[(df["Day"] >= start_day) & (df["Day"] <= end_day)]
            selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="manual_range_cols")

            if selected_cols:
             try:
                 selected_range = range_df[selected_cols]

                 if selected_range.empty:
                  st.warning("⚠️ No data in selected range.")
                 else:
                   st.write("📊 Totals:", selected_range.sum().round(2))
                   st.write("📈 Averages:", selected_range.mean().round(2))
                   st.write("🔺 Max:", selected_range.max().round(2))
                   st.write("🔻 Min:", selected_range.min().round(2))
                   st.line_chart(selected_range)
             except KeyError as e:
                   st.error(f"🚫 Invalid column selection: {e}")
             except Exception as e:
                   st.error(f"💥 Unexpected error: {e}")
            else:
                   st.info("Choose at least one column to view stats.")


# === Tab 4 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)
with tabs[3]:
   
 st.title("📈 Jemey Live Market Dashboard")

 st.sidebar.header("📊 Market Settings")
 asset_type = st.sidebar.radio("Select Asset Type", ["Crypto", "Stock"])
 symbol = st.sidebar.selectbox("Choose Symbol", ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"])
 lookback = st.sidebar.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

 @st.cache_data
 def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.reset_index(inplace=True)

    # Flatten column names if needed (only if multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Rename columns if they have ticker prefix
    rename_map = {}
    for col in df.columns:
        if f" {symbol}" in col:
            rename_map[col] = col.replace(f" {symbol}", "")
    df.rename(columns=rename_map, inplace=True)

    return df

 try:
     df = fetch_market_data(symbol, period=lookback)

     # Check required columns
     required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
     if not all(col in df.columns for col in required_columns):
      st.warning("Trying alternative fallback column structure...")
     df.rename(columns={
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
        }, inplace=True)

     # Validate again
     if not all(col in df.columns for col in required_columns):
      raise ValueError("One or more required columns are missing in the data.")

     df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
     df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
     df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
     df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
     df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
     df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
     df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
     df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
     df['MACD_Hist'] = df['MACD'] - df['Signal']

     st.subheader(f"📊 Chart for {symbol} ({lookback})")

     fig = go.Figure()
     fig.add_trace(go.Candlestick(
       x=df['Date'],
       open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
       name='Candles'))
     for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
      fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))

     fig.update_layout(title='Price Chart + EMA', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
     st.plotly_chart(fig, use_container_width=True)

     st.markdown("---")
     st.subheader("📉 RSI")
     fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
     fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
     fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
     st.plotly_chart(fig_rsi, use_container_width=True)

     st.subheader("📈 MACD")
     fig_macd = go.Figure()
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
     fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
     st.plotly_chart(fig_macd, use_container_width=True)

     st.subheader("📊 ATR - Volatility")
     fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
     st.plotly_chart(fig_atr, use_container_width=True)

     with st.expander("💡 Risk Management Guidelines"):
         st.markdown("""
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

     st.markdown("---")
     st.subheader("📂 Raw Market Data")
     st.dataframe(df)

     st.markdown("---")
     st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

 except Exception as e:
     st.error(f"⚠️ Failed to fetch market data: {e}")


 st.markdown("---")
 st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
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
    "📊 Jemey Real-Time Dashboard"
])

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

    # Parse schedule
    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    # Custom names and ratios
    names = []
    ratios = []
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

            while recovered < goal:
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
            st.success(f"✅ Full Recovery Achieved in {day - 1} days")

            share_cols = [f"{name} Share" for name in names]
            st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
            st.dataframe(df)

            # Manual day range inputs
            st.markdown("### 🎯 Analyze by Manual Day Range")
            max_day = int(df["Day"].max())
            start_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="start_day_input")
            end_day = st.number_input("End Day", min_value=start_day, max_value=max_day, value=max_day, key="end_day_input")

            range_df = df[(df["Day"] >= start_day) & (df["Day"] <= end_day)]
            selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="manual_range_cols")

            if selected_cols:
             try:
                 selected_range = range_df[selected_cols]

                 if selected_range.empty:
                  st.warning("⚠️ No data in selected range.")
                 else:
                   st.write("📊 Totals:", selected_range.sum().round(2))
                   st.write("📈 Averages:", selected_range.mean().round(2))
                   st.write("🔺 Max:", selected_range.max().round(2))
                   st.write("🔻 Min:", selected_range.min().round(2))
                   st.line_chart(selected_range)
             except KeyError as e:
                   st.error(f"🚫 Invalid column selection: {e}")
             except Exception as e:
                   st.error(f"💥 Unexpected error: {e}")
            else:
                   st.info("Choose at least one column to view stats.")


# === Tab 4 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)
with tabs[3]:
   
 st.title("📈 Jemey Live Market Dashboard")

 st.sidebar.header("📊 Market Settings")
 asset_type = st.sidebar.radio("Select Asset Type", ["Crypto", "Stock"])
 symbol = st.sidebar.selectbox("Choose Symbol", ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"])
 lookback = st.sidebar.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

 @st.cache_data
 def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.reset_index(inplace=True)

    # Flatten column names if needed (only if multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Rename columns if they have ticker prefix
    rename_map = {}
    for col in df.columns:
        if f" {symbol}" in col:
            rename_map[col] = col.replace(f" {symbol}", "")
    df.rename(columns=rename_map, inplace=True)

    return df

 try:
     df = fetch_market_data(symbol, period=lookback)

     # Check required columns
     required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
     if not all(col in df.columns for col in required_columns):
      st.warning("Trying alternative fallback column structure...")
     df.rename(columns={
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
        }, inplace=True)

     # Validate again
     if not all(col in df.columns for col in required_columns):
      raise ValueError("One or more required columns are missing in the data.")

     df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
     df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
     df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
     df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
     df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
     df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
     df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
     df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
     df['MACD_Hist'] = df['MACD'] - df['Signal']

     st.subheader(f"📊 Chart for {symbol} ({lookback})")

     fig = go.Figure()
     fig.add_trace(go.Candlestick(
       x=df['Date'],
       open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
       name='Candles'))
     for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
      fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))

     fig.update_layout(title='Price Chart + EMA', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
     st.plotly_chart(fig, use_container_width=True)

     st.markdown("---")
     st.subheader("📉 RSI")
     fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
     fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
     fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
     st.plotly_chart(fig_rsi, use_container_width=True)

     st.subheader("📈 MACD")
     fig_macd = go.Figure()
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
     fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
     st.plotly_chart(fig_macd, use_container_width=True)

     st.subheader("📊 ATR - Volatility")
     fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
     st.plotly_chart(fig_atr, use_container_width=True)

     with st.expander("💡 Risk Management Guidelines"):
         st.markdown("""
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

     st.markdown("---")
     st.subheader("📂 Raw Market Data")
     st.dataframe(df)

     st.markdown("---")
     st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

 except Exception as e:
     st.error(f"⚠️ Failed to fetch market data: {e}")


 st.markdown("---")
 st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
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
    "📊 Jemey Real-Time Dashboard"
])

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

    # Parse schedule
    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    # Custom names and ratios
    names = []
    ratios = []
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

            while recovered < goal:
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
            st.success(f"✅ Full Recovery Achieved in {day - 1} days")

            share_cols = [f"{name} Share" for name in names]
            st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
            st.dataframe(df)

            # Manual day range inputs
            st.markdown("### 🎯 Analyze by Manual Day Range")
            max_day = int(df["Day"].max())
            start_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="start_day_input")
            end_day = st.number_input("End Day", min_value=start_day, max_value=max_day, value=max_day, key="end_day_input")

            range_df = df[(df["Day"] >= start_day) & (df["Day"] <= end_day)]
            selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="manual_range_cols")

            if selected_cols:
             try:
                 selected_range = range_df[selected_cols]

                 if selected_range.empty:
                  st.warning("⚠️ No data in selected range.")
                 else:
                   st.write("📊 Totals:", selected_range.sum().round(2))
                   st.write("📈 Averages:", selected_range.mean().round(2))
                   st.write("🔺 Max:", selected_range.max().round(2))
                   st.write("🔻 Min:", selected_range.min().round(2))
                   st.line_chart(selected_range)
             except KeyError as e:
                   st.error(f"🚫 Invalid column selection: {e}")
             except Exception as e:
                   st.error(f"💥 Unexpected error: {e}")
            else:
                   st.info("Choose at least one column to view stats.")


# === Tab 4 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)
with tabs[3]:
   
 st.title("📈 Jemey Live Market Dashboard")

 st.sidebar.header("📊 Market Settings")
 asset_type = st.sidebar.radio("Select Asset Type", ["Crypto", "Stock"])
 symbol = st.sidebar.selectbox("Choose Symbol", ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"])
 lookback = st.sidebar.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

 @st.cache_data
 def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.reset_index(inplace=True)

    # Flatten column names if needed (only if multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Rename columns if they have ticker prefix
    rename_map = {}
    for col in df.columns:
        if f" {symbol}" in col:
            rename_map[col] = col.replace(f" {symbol}", "")
    df.rename(columns=rename_map, inplace=True)

    return df

 try:
     df = fetch_market_data(symbol, period=lookback)

     # Check required columns
     required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
     if not all(col in df.columns for col in required_columns):
      st.warning("Trying alternative fallback column structure...")
     df.rename(columns={
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
        }, inplace=True)

     # Validate again
     if not all(col in df.columns for col in required_columns):
      raise ValueError("One or more required columns are missing in the data.")

     df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
     df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
     df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
     df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
     df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
     df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
     df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
     df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
     df['MACD_Hist'] = df['MACD'] - df['Signal']

     st.subheader(f"📊 Chart for {symbol} ({lookback})")

     fig = go.Figure()
     fig.add_trace(go.Candlestick(
       x=df['Date'],
       open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
       name='Candles'))
     for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
      fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))

     fig.update_layout(title='Price Chart + EMA', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
     st.plotly_chart(fig, use_container_width=True)

     st.markdown("---")
     st.subheader("📉 RSI")
     fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
     fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
     fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
     st.plotly_chart(fig_rsi, use_container_width=True)

     st.subheader("📈 MACD")
     fig_macd = go.Figure()
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
     fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
     st.plotly_chart(fig_macd, use_container_width=True)

     st.subheader("📊 ATR - Volatility")
     fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
     st.plotly_chart(fig_atr, use_container_width=True)

     with st.expander("💡 Risk Management Guidelines"):
         st.markdown("""
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

     st.markdown("---")
     st.subheader("📂 Raw Market Data")
     st.dataframe(df)

     st.markdown("---")
     st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

 except Exception as e:
     st.error(f"⚠️ Failed to fetch market data: {e}")


 st.markdown("---")
 st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
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
    "📊 Jemey Real-Time Dashboard"
])

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

    # Parse schedule
    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    # Custom names and ratios
    names = []
    ratios = []
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

            while recovered < goal:
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
            st.success(f"✅ Full Recovery Achieved in {day - 1} days")

            share_cols = [f"{name} Share" for name in names]
            st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
            st.dataframe(df)

            # Manual day range inputs
            st.markdown("### 🎯 Analyze by Manual Day Range")
            max_day = int(df["Day"].max())
            start_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="start_day_input")
            end_day = st.number_input("End Day", min_value=start_day, max_value=max_day, value=max_day, key="end_day_input")

            range_df = df[(df["Day"] >= start_day) & (df["Day"] <= end_day)]
            selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="manual_range_cols")

            if selected_cols:
             try:
                 selected_range = range_df[selected_cols]

                 if selected_range.empty:
                  st.warning("⚠️ No data in selected range.")
                 else:
                   st.write("📊 Totals:", selected_range.sum().round(2))
                   st.write("📈 Averages:", selected_range.mean().round(2))
                   st.write("🔺 Max:", selected_range.max().round(2))
                   st.write("🔻 Min:", selected_range.min().round(2))
                   st.line_chart(selected_range)
             except KeyError as e:
                   st.error(f"🚫 Invalid column selection: {e}")
             except Exception as e:
                   st.error(f"💥 Unexpected error: {e}")
            else:
                   st.info("Choose at least one column to view stats.")


# === Tab 4 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)
with tabs[3]:
   
 st.title("📈 Jemey Live Market Dashboard")

 st.sidebar.header("📊 Market Settings")
 asset_type = st.sidebar.radio("Select Asset Type", ["Crypto", "Stock"])
 symbol = st.sidebar.selectbox("Choose Symbol", ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"])
 lookback = st.sidebar.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

 @st.cache_data
 def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.reset_index(inplace=True)

    # Flatten column names if needed (only if multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Rename columns if they have ticker prefix
    rename_map = {}
    for col in df.columns:
        if f" {symbol}" in col:
            rename_map[col] = col.replace(f" {symbol}", "")
    df.rename(columns=rename_map, inplace=True)

    return df

 try:
     df = fetch_market_data(symbol, period=lookback)

     # Check required columns
     required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
     if not all(col in df.columns for col in required_columns):
      st.warning("Trying alternative fallback column structure...")
     df.rename(columns={
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
        }, inplace=True)

     # Validate again
     if not all(col in df.columns for col in required_columns):
      raise ValueError("One or more required columns are missing in the data.")

     df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
     df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
     df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
     df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
     df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
     df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
     df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
     df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
     df['MACD_Hist'] = df['MACD'] - df['Signal']

     st.subheader(f"📊 Chart for {symbol} ({lookback})")

     fig = go.Figure()
     fig.add_trace(go.Candlestick(
       x=df['Date'],
       open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
       name='Candles'))
     for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
      fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))

     fig.update_layout(title='Price Chart + EMA', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
     st.plotly_chart(fig, use_container_width=True)

     st.markdown("---")
     st.subheader("📉 RSI")
     fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
     fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
     fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
     st.plotly_chart(fig_rsi, use_container_width=True)

     st.subheader("📈 MACD")
     fig_macd = go.Figure()
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
     fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
     st.plotly_chart(fig_macd, use_container_width=True)

     st.subheader("📊 ATR - Volatility")
     fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
     st.plotly_chart(fig_atr, use_container_width=True)

     with st.expander("💡 Risk Management Guidelines"):
         st.markdown("""
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

     st.markdown("---")
     st.subheader("📂 Raw Market Data")
     st.dataframe(df)

     st.markdown("---")
     st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

 except Exception as e:
     st.error(f"⚠️ Failed to fetch market data: {e}")


 st.markdown("---")
 st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
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
    "📊 Jemey Real-Time Dashboard"
])

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

    # Parse schedule
    schedule = []
    for s in schedule_str.split(","):
        if ":" in s:
            range_part, rate = s.split(":")
            start, end = map(int, range_part.split("-"))
            schedule.append({"from": start, "to": end, "rate": float(rate)})

    # Custom names and ratios
    names = []
    ratios = []
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

            while recovered < goal:
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
            st.success(f"✅ Full Recovery Achieved in {day - 1} days")

            share_cols = [f"{name} Share" for name in names]
            st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
            st.dataframe(df)

            # Manual day range inputs
            st.markdown("### 🎯 Analyze by Manual Day Range")
            max_day = int(df["Day"].max())
            start_day = st.number_input("Start Day", min_value=1, max_value=max_day, value=1, key="start_day_input")
            end_day = st.number_input("End Day", min_value=start_day, max_value=max_day, value=max_day, key="end_day_input")

            range_df = df[(df["Day"] >= start_day) & (df["Day"] <= end_day)]
            selected_cols = st.multiselect("Choose People to Analyze", share_cols, default=share_cols, key="manual_range_cols")

            if selected_cols:
             try:
                 selected_range = range_df[selected_cols]

                 if selected_range.empty:
                  st.warning("⚠️ No data in selected range.")
                 else:
                   st.write("📊 Totals:", selected_range.sum().round(2))
                   st.write("📈 Averages:", selected_range.mean().round(2))
                   st.write("🔺 Max:", selected_range.max().round(2))
                   st.write("🔻 Min:", selected_range.min().round(2))
                   st.line_chart(selected_range)
             except KeyError as e:
                   st.error(f"🚫 Invalid column selection: {e}")
             except Exception as e:
                   st.error(f"💥 Unexpected error: {e}")
            else:
                   st.info("Choose at least one column to view stats.")


# === Tab 4 ===

# 📊 Trading Dashboard inside tab3 (with sidebar layout preserved)
with tabs[3]:
   
 st.title("📈 Jemey Live Market Dashboard")

 st.sidebar.header("📊 Market Settings")
 asset_type = st.sidebar.radio("Select Asset Type", ["Crypto", "Stock"])
 symbol = st.sidebar.selectbox("Choose Symbol", ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"])
 lookback = st.sidebar.selectbox("Lookback Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

 @st.cache_data
 def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1d"):
    df = yf.download(tickers=symbol, period=period, interval=interval)
    df.reset_index(inplace=True)

    # Flatten column names if needed (only if multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Rename columns if they have ticker prefix
    rename_map = {}
    for col in df.columns:
        if f" {symbol}" in col:
            rename_map[col] = col.replace(f" {symbol}", "")
    df.rename(columns=rename_map, inplace=True)

    return df

 try:
     df = fetch_market_data(symbol, period=lookback)

     # Check required columns
     required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
     if not all(col in df.columns for col in required_columns):
      st.warning("Trying alternative fallback column structure...")
     df.rename(columns={
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
        }, inplace=True)

     # Validate again
     if not all(col in df.columns for col in required_columns):
      raise ValueError("One or more required columns are missing in the data.")

     df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
     df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
     df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
     df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
     df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
     df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
     df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
     df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
     df['MACD_Hist'] = df['MACD'] - df['Signal']

     st.subheader(f"📊 Chart for {symbol} ({lookback})")

     fig = go.Figure()
     fig.add_trace(go.Candlestick(
       x=df['Date'],
       open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
       name='Candles'))
     for ema in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
      fig.add_trace(go.Scatter(x=df['Date'], y=df[ema], mode='lines', name=ema))

     fig.update_layout(title='Price Chart + EMA', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
     st.plotly_chart(fig, use_container_width=True)

     st.markdown("---")
     st.subheader("📉 RSI")
     fig_rsi = px.line(df, x='Date', y='RSI', title='RSI')
     fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
     fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
     st.plotly_chart(fig_rsi, use_container_width=True)

     st.subheader("📈 MACD")
     fig_macd = go.Figure()
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
     fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], mode='lines', name='Signal Line'))
     fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram'))
     st.plotly_chart(fig_macd, use_container_width=True)

     st.subheader("📊 ATR - Volatility")
     fig_atr = px.line(df, x='Date', y='ATR', title='ATR')
     st.plotly_chart(fig_atr, use_container_width=True)

     with st.expander("💡 Risk Management Guidelines"):
         st.markdown("""
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

     st.markdown("---")
     st.subheader("📂 Raw Market Data")
     st.dataframe(df)

     st.markdown("---")
     st.caption("📘 Disclaimer: This analysis is for informational purposes only. Trading involves risk. Past performance does not guarantee future results.")

 except Exception as e:
     st.error(f"⚠️ Failed to fetch market data: {e}")


 st.markdown("---")
 st.markdown("<center>Built with ❤️ by Ahmed & Jemey — Powered by Live Market Data</center>", unsafe_allow_html=True)





# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
