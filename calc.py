import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from fpdf import FPDF

# -----------------------
# Compound Growth Function
# -----------------------

def compound_growth(principal, rate, days):
    rate_decimal = rate / 100
    data = []
    amount = principal
    total_profit = 0

    for day in range(1, days + 1):
        profit = amount * rate_decimal
        amount += profit
        total_profit += profit
        data.append({
            "Day": day,
            "Daily Profit": round(profit, 2),
            "Total Amount": round(amount, 2)
        })

    df = pd.DataFrame(data)
    return df, total_profit, amount

# -----------------------
# Daily Withdrawal Logic
# -----------------------

def simulate_withdrawals(principal, rate, days):
    rate_decimal = rate / 100
    amount = principal
    withdrawn_total = 0
    data = []

    for day in range(1, days + 1):
        profit = amount * rate_decimal
        withdrawal = profit if withdrawn_total + profit <= principal else max(0, principal - withdrawn_total)
        amount += profit - withdrawal
        withdrawn_total += withdrawal
        data.append({
            "Day": day,
            "Withdrawn": round(withdrawal, 2),
            "Balance": round(amount, 2)
        })

    df = pd.DataFrame(data)
    return df, withdrawn_total, amount

# -----------------------
# Export to Excel
# -----------------------

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Growth')
    return output.getvalue()

# -----------------------
# Export to PDF
# -----------------------

def generate_pdf_report(principal, rate, days, final_amount, total_profit):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Compound Interest Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Initial Investment: ${principal:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Daily Interest Rate: {rate}%", ln=True)
    pdf.cell(200, 10, txt=f"Number of Days: {days}", ln=True)
    pdf.cell(200, 10, txt=f"Final Amount: ${final_amount:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Total Profit: ${total_profit:,.2f}", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin1')  # FPDF returns str in latin1
    return BytesIO(pdf_bytes)

# -----------------------
# UI
# -----------------------

st.set_page_config(page_title="Smart Compound Calculator", layout="centered")

# Theme toggle
theme = st.radio("🌓 Select Theme", ["Light", "Dark"], horizontal=True)
if theme == "Dark":
    st.markdown("<style>body { background-color: #121212; color: #fafafa; }</style>", unsafe_allow_html=True)

st.title("📈 Smart Compound Interest Calculator")

currency = st.selectbox("Select Currency", ["$", "€", "£", "EGP", "AED", "SAR"], index=0)
initial = st.number_input(f"Initial Investment ({currency})", value=500.0, step=100.0)
days = st.number_input("Number of Days", value=1, step=1)
rate = st.number_input("Daily Interest Rate (%)", value=2.217, step=0.1)

if st.button("🔍 Calculate"):
    df, profit, final_amount = compound_growth(initial, rate, days)

    st.success(f"💵 Total Profit: {currency}{profit:,.2f}")
    st.success(f"📦 Resulting Amount: {currency}{final_amount:,.2f}")

    # Chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Day"], df["Total Amount"], label="Total Value", color='green')
    ax.fill_between(df["Day"], df["Total Amount"], alpha=0.3)
    ax.set_xlabel("Day")
    ax.set_ylabel(f"Amount ({currency})")
    ax.set_title("Compound Growth Over Time")
    ax.legend()
    st.pyplot(fig)

    # Excel export
    excel_data = convert_df_to_excel(df)
    st.download_button("⬇️ Download Excel Report", data=excel_data, file_name="compound_growth.xlsx")

    # PDF export
    pdf_bytes = generate_pdf_report(initial, rate, days, final_amount, profit)
    st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="compound_report.pdf")

    # Daily withdrawal logic
    st.markdown("### 💸 Simulated Daily Withdrawals")
    withdraw_df, total_withdrawn, final_balance = simulate_withdrawals(initial, rate, days)
    st.info(f"Recovered: {currency}{total_withdrawn:,.2f} (Original: {currency}{initial:,.2f})")
    st.dataframe(withdraw_df)

    # (Optional) Email sending placeholder
    with st.expander("📧 Email Report (Coming Soon)"):
        st.text_input("Enter your email address")
        st.button("Send (Not Active Yet)")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Ahmed Elmorsy | Everything is editable on my PC")
