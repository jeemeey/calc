# Smart Compound + Recovery + Sharing
import streamlit as st
import pandas as pd
from io import BytesIO
import smtplib
from email.message import EmailMessage
import os
import datetime
from PIL import Image
import re



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
    "📊 Chart Analyzer"
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
        invest = st.number_input("Initial Investment", value=10000.0, step=100.0)
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
with tabs[3]:
    st.subheader("📊 Pro Price Trend Analyzer (Manual Input + Full Analysis)")

    st.markdown("""
    Paste your price points (from TradingView, Binance, etc) and get full pro analysis:
    - 🔼 Count of Up/Down Moves
    - 📈 Biggest Surge & 📉 Biggest Drop
    - 📊 Volatility Score (Standard Deviation)
    - 🪜 Consolidation Detector (range-bound windows)
    - 🔁 Reversal detection
    - 📆 TP/SL Time Tracker
    - 🔁 Double Top / Bottom Pattern Detection
    """)

    price_input = st.text_area("Paste price list (comma-separated or line-by-line)", "83000, 82500, 82000, 82700, 83400, 82900, 82100, 81000")

    if price_input:
        raw_prices = [s.strip() for s in price_input.replace("\n", ",").split(",")]
        try:
            prices = [float(p.replace(",", "")) for p in raw_prices if float(p.replace(",", "")) > 1000]
            if len(prices) < 2:
                st.warning("Please enter at least two valid prices.")
            else:
                import numpy as np

                # 📊 Basic Trend Count
                up_moves = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
                down_moves = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i - 1])

                # 🔁 Longest Streaks
                up_streak = down_streak = max_up = max_down = 0
                for p1, p2 in zip(prices, prices[1:]):
                    if p2 > p1:
                        up_streak += 1
                        down_streak = 0
                    elif p2 < p1:
                        down_streak += 1
                        up_streak = 0
                    else:
                        up_streak = down_streak = 0
                    max_up = max(max_up, up_streak)
                    max_down = max(max_down, down_streak)

                # 🔁 Reversal Detection
                reversals = sum(1 for i in range(2, len(prices)) if (prices[i] - prices[i-1]) * (prices[i-1] - prices[i-2]) < 0)

                # 🚀 Surge & Drop
                max_drop = max(
                              [(prices[i] - prices[j], prices[j], prices[i], j, i)
                              for i in range(len(prices)) for j in range(i)
                              if prices[j] > prices[i]],
                              default=(0, 0, 0, 0, 0)
                              )

                max_surge = max(
                                [(prices[i] - prices[j], prices[j], prices[i], j, i)
                                for i in range(len(prices)) for j in range(i)
                                if prices[j] < prices[i]],
                                default=(0, 0, 0, 0, 0)
                                )


                # 📊 Volatility (Standard Deviation)
                std_dev = round(np.std(prices), 2)

                # 🪜 Consolidation Zones
                consolidations = []
                window = 5
                for i in range(len(prices) - window):
                    sub = prices[i:i + window]
                    if max(sub) - min(sub) < 0.01 * np.mean(sub):
                        consolidations.append((i, i + window - 1))

                # 📆 TP/SL Simulation (assume +5% TP, -3% SL from point 0)
                tp_target = prices[0] * 1.05
                sl_target = prices[0] * 0.97
                tp_hit = sl_hit = None
                for i, price in enumerate(prices):
                    if tp_hit is None and price >= tp_target:
                        tp_hit = i
                    if sl_hit is None and price <= sl_target:
                        sl_hit = i

                # 🔁 Double Top / Bottom Detection
                patterns = []
                for i in range(1, len(prices) - 2):
                    if prices[i - 1] < prices[i] > prices[i + 1] and abs(prices[i] - prices[i + 2]) < 0.01 * prices[i]:
                        patterns.append(f"Double Top near index {i}")
                    if prices[i - 1] > prices[i] < prices[i + 1] and abs(prices[i] - prices[i + 2]) < 0.01 * prices[i]:
                        patterns.append(f"Double Bottom near index {i}")

                st.success("✅ Full Pro Analysis Complete")

                st.markdown(f"""
                ### 🧠 Summary
                - 🔼 Up moves: **{up_moves}**
                - 🔽 Down moves: **{down_moves}**
                - 🔁 Reversals detected: **{reversals}**
                - 📈 Longest uptrend: **{max_up + 1} steps**
                - 📉 Longest downtrend: **{max_down + 1} steps**
                - 🚀 Biggest Surge: **+{max_surge[0]:,.2f}** from {max_surge[1]} → {max_surge[2]} (pts {max_surge[3]} → {max_surge[4]})
                - 💥 Biggest Drop: **-{abs(max_drop[0]):,.2f}** from {max_drop[1]} → {max_drop[2]} (pts {max_drop[3]} → {max_drop[4]})
                - 📊 Volatility (std dev): **{std_dev}**
                - 🪜 Consolidation zones: **{len(consolidations)}**
                - 🎯 TP Hit at: **step {tp_hit}**  |  🛑 SL Hit at: **step {sl_hit}**
                - 🔁 Patterns found: **{len(patterns)}**
                """)

                if patterns:
                    st.markdown("#### 📌 Pattern Details")
                    for p in patterns:
                        st.markdown(f"- {p}")

                st.line_chart(prices)

        except Exception as e:
            st.error(f"Error processing input: {e}")




 

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com") 
