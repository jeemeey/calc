# Smart Compound + Recovery + Sharing
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import smtplib
from email.message import EmailMessage
import os
import datetime

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
tabs = st.tabs(["📈 Compound Calculator", "💸 Withdrawal Simulator", "🤝 Shared Investment Plan"])

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
    st.subheader("🤝 Shared Strategy")
    col1, col2 = st.columns(2)
    with col1:
        s_initinv = st.number_input("Initial Investment", 500.0, step=50.0, key="init_inv_1")
        sched_str = st.text_area("Gain Schedule", "1-10:3.3,11-20:2.2,21-30:4.4")
        people = st.number_input("People Sharing Profit", 2, step=1)
    with col2:
        split = st.number_input("Recovery Split Ratio (e.g. 0.5 = 50% to recovery)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    sched = []
    for s in sched_str.split(","):
        if ":" in s:
            rng, pct = s.split(":")
            frm, to = map(int, rng.split("-"))
            sched.append({"from": frm, "to": to, "rate": float(pct)})

    if st.button("▶️ Run Shared Strategy"):
        df, recov, days = shared_strategy(s_initinv, sched, split, people)
        st.success(f"✅ Recovery in {days} days")
        share_cols = [f"Person {i} Share" for i in range(1, people + 1)]
        st.line_chart(df.set_index("Day")[["Recovered"] + share_cols])
        st.dataframe(df)
        st.download_button("⬇️ Download Excel", data=convert_df_to_excel(df), file_name="shared_plan.xlsx")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Engineer / Ahmed Elmorsy | Everything is editable on my laptop "
"For any Request or Suggestion feel free | contact me anytime on Gmail: Jemey.Embeddedsys@Gmail.com")
