import math

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI


def proportions_ztest(count, nobs):
    """Two-proportion z-test (two-tailed), equivalent to scipy.stats.proportions_ztest."""
    conv_a, conv_b = count
    users_a, users_b = nobs
    p_pool = (conv_a + conv_b) / (users_a + users_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / users_a + 1 / users_b))
    z_stat = (conv_b / users_b - conv_a / users_a) / se
    p_value = math.erfc(abs(z_stat) / math.sqrt(2))
    return z_stat, p_value


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="A/B Testing Agent", layout="wide")
st.title("A/B Testing Agent")
st.markdown("Upload your experiment data to get statistical insights and a recommendation.")

# ── Sidebar: AI configuration ─────────────────────────────────────────────────
with st.sidebar:
    st.header("AI Configuration")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored or logged. Required for AI features.",
    )
    if openai_api_key:
        st.success("API key provided.")
    else:
        st.info("Enter your OpenAI API key to unlock AI-powered analysis.")

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
REQUIRED_COLUMNS = {"variant", "users", "conversions"}


# ── Validation ────────────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame) -> str | None:
    """Return an error message string, or None if valid."""
    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing:
        return f"Missing required columns: {', '.join(missing)}. Expected: variant, users, conversions."
    df.columns = df.columns.str.lower()
    variants = df["variant"].str.upper().unique().tolist()
    if "A" not in variants or "B" not in variants:
        return f"Expected variants 'A' and 'B', but found: {variants}."
    for col in ["users", "conversions"]:
        if not pd.to_numeric(df[col], errors="coerce").notna().all():
            return f"Column '{col}' must contain numeric values only."
    return None


# ── Metrics computation ───────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df["variant"] = df["variant"].str.upper()

    agg = df.groupby("variant")[["users", "conversions"]].sum()
    users_a, conv_a = int(agg.loc["A", "users"]), int(agg.loc["A", "conversions"])
    users_b, conv_b = int(agg.loc["B", "users"]), int(agg.loc["B", "conversions"])

    rate_a = conv_a / users_a
    rate_b = conv_b / users_b
    abs_diff = rate_b - rate_a
    uplift = ((rate_b - rate_a) / rate_a) * 100 if rate_a > 0 else 0.0

    z_stat, p_value = proportions_ztest(
        count=[conv_a, conv_b],
        nobs=[users_a, users_b],
    )

    return {
        "users_a": users_a,
        "users_b": users_b,
        "conv_a": conv_a,
        "conv_b": conv_b,
        "rate_a": rate_a,
        "rate_b": rate_b,
        "abs_diff": abs_diff,
        "uplift": uplift,
        "z_stat": z_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "confidence": (1 - p_value) * 100,
    }


# ── Recommendation ────────────────────────────────────────────────────────────
def generate_recommendation(m: dict) -> tuple:
    """Return (recommendation_text, streamlit_widget_type)."""
    better = "B" if m["rate_b"] > m["rate_a"] else "A"

    rate_a_pct = m["rate_a"] * 100
    rate_b_pct = m["rate_b"] * 100
    uplift     = abs(m["uplift"])
    p_val      = m["p_value"]
    conf       = min(m["confidence"], 99.9)

    if m["significant"]:
        if better == "B":
            text = (
                f"**Ship B.** Variant B outperforms A with a conversion rate of "
                f"{rate_b_pct:.2f}% vs {rate_a_pct:.2f}% — a {uplift:.1f}% uplift. "
                f"The result is statistically significant (p = {p_val:.4f}, {conf:.1f}% confidence), "
                f"meaning it is very unlikely to be due to chance. "
                f"Recommend rolling out Variant B to all users."
            )
            return text, "success"
        else:
            text = (
                f"**Keep A.** Variant A outperforms B with a conversion rate of "
                f"{rate_a_pct:.2f}% vs {rate_b_pct:.2f}%. "
                f"The result is statistically significant (p = {p_val:.4f}, {conf:.1f}% confidence). "
                f"Variant B underperforms — do not ship it."
            )
            return text, "error"
    else:
        text = (
            f"**Run the test longer.** Variant {better} currently shows a slightly higher "
            f"conversion rate ({rate_b_pct:.2f}% vs {rate_a_pct:.2f}%), but the result is "
            f"**not statistically significant** (p = {p_val:.4f}, {conf:.1f}% confidence). "
            f"There is insufficient evidence to declare a winner. "
            f"Collect more data before making a decision."
        )
        return text, "warning"


# ── AI: Executive summary ─────────────────────────────────────────────────────
def generate_exec_summary(m: dict, api_key: str) -> str:
    """One-sentence plain-English summary for executives."""
    user_prompt = (
        "A/B test results:\n"
        f"- Variant A: {m['users_a']:,} users, {m['conv_a']:,} conversions, "
        f"{m['rate_a'] * 100:.2f}% conversion rate\n"
        f"- Variant B: {m['users_b']:,} users, {m['conv_b']:,} conversions, "
        f"{m['rate_b'] * 100:.2f}% conversion rate\n"
        f"- Uplift: {m['uplift']:+.1f}%\n"
        f"- p-value: {m['p_value']:.4f}\n"
        f"- Statistically significant: {m['significant']}\n"
        f"- Confidence: {min(m['confidence'], 99.9):.1f}%\n"
        "Summarize the result in exactly one plain-English sentence."
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=80,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise data analyst. Write exactly one plain-English sentence "
                    "summarizing the outcome of an A/B test. No markdown, no bullet points. "
                    "Be specific with numbers. Suitable for an executive reading in 5 seconds."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ── AI: Narrative analysis ────────────────────────────────────────────────────
def generate_narrative(m: dict, rec_text: str, api_key: str) -> str:
    """Full markdown business analysis of the A/B test results."""
    user_prompt = (
        "Analyze the following A/B test results and write a comprehensive business narrative.\n\n"
        "## Test Data\n"
        f"- Variant A: {m['users_a']:,} users, {m['conv_a']:,} conversions "
        f"({m['rate_a'] * 100:.2f}% rate)\n"
        f"- Variant B: {m['users_b']:,} users, {m['conv_b']:,} conversions "
        f"({m['rate_b'] * 100:.2f}% rate)\n"
        f"- Absolute difference: {m['abs_diff'] * 100:+.2f} percentage points\n"
        f"- Relative uplift: {m['uplift']:+.1f}%\n"
        f"- Z-statistic: {m['z_stat']:.4f}\n"
        f"- p-value: {m['p_value']:.4f}\n"
        f"- Confidence: {min(m['confidence'], 99.9):.1f}%\n"
        f"- Statistically significant at 95% threshold: {m['significant']}\n\n"
        "## Template Recommendation\n"
        f"{rec_text}\n\n"
        "## Cover These Sections\n"
        "1. **What the results mean** — interpret the numbers in plain English\n"
        "2. **Business implications** — revenue impact, user experience, strategic fit\n"
        "3. **Statistical caveats** — what the p-value does and does not tell us, "
        "sample size considerations, potential biases\n"
        "4. **Risks** — what could go wrong if the recommendation is followed\n"
        "5. **Recommended next steps** — concrete, prioritized actions\n"
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior data scientist and business strategist writing an A/B test "
                    "analysis report for a product team. Write in clear, professional prose using "
                    "markdown formatting (headers, bold, bullet points where appropriate). "
                    "Do not use jargon without explanation. Your analysis should be actionable."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ── AI: Follow-up Q&A ─────────────────────────────────────────────────────────
def answer_ab_question(
    question: str,
    m: dict,
    chat_history: list,
    api_key: str,
) -> str:
    """Answer a follow-up question about the A/B test results."""
    system_prompt = (
        "You are an expert A/B testing analyst and statistician. Answer questions about "
        "the following experiment concisely and accurately. Always ground your answers in "
        "the specific numbers provided. If a question is outside the scope of the test data, "
        "say so clearly.\n\n"
        "## Experiment Data\n"
        f"- Variant A: {m['users_a']:,} users, {m['conv_a']:,} conversions, "
        f"{m['rate_a'] * 100:.2f}% conversion rate\n"
        f"- Variant B: {m['users_b']:,} users, {m['conv_b']:,} conversions, "
        f"{m['rate_b'] * 100:.2f}% conversion rate\n"
        f"- Absolute difference: {m['abs_diff'] * 100:+.2f} percentage points\n"
        f"- Relative uplift: {m['uplift']:+.1f}%\n"
        f"- Z-statistic: {m['z_stat']:.4f}, p-value: {m['p_value']:.4f}\n"
        f"- Confidence: {min(m['confidence'], 99.9):.1f}%\n"
        f"- Statistically significant: {m['significant']}\n"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=400,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


# ── File loading ──────────────────────────────────────────────────────────────
def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file, engine="openpyxl")


# ── Main UI ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your A/B test data (CSV or Excel)",
    type=["csv", "xlsx"],
)

if uploaded_file is not None:
    if len(uploaded_file.getvalue()) > MAX_FILE_SIZE_BYTES:
        st.error(
            f"File too large ({len(uploaded_file.getvalue()) / 1024 / 1024:.1f} MB). "
            "Maximum allowed size is 10 MB."
        )
        st.stop()

    try:
        df = load_file(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # ── Preview ───────────────────────────────────────────────────────────────
    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)

    # ── Validate ──────────────────────────────────────────────────────────────
    error = validate_dataframe(df)
    if error:
        st.error(error)
        st.markdown(
            "**Expected format:**\n```\nvariant, users, conversions\nA, 1000, 120\nB, 1000, 145\n```"
        )
        st.stop()

    # ── Compute ───────────────────────────────────────────────────────────────
    try:
        m = compute_metrics(df)
    except Exception as e:
        st.error(f"Error computing metrics: {e}")
        st.stop()

    # ── Invalidate AI caches when a new file is uploaded ──────────────────────
    if st.session_state.get("_ai_file") != uploaded_file.name:
        st.session_state["_ai_file"] = uploaded_file.name
        for key in ["ai_exec_summary", "ai_narrative", "chat_history"]:
            st.session_state.pop(key, None)

    # ── Metrics display ───────────────────────────────────────────────────────
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conversion Rate — A", f"{m['rate_a'] * 100:.2f}%",
                help=f"{m['conv_a']} conversions / {m['users_a']} users")
    col2.metric("Conversion Rate — B", f"{m['rate_b'] * 100:.2f}%",
                delta=f"{m['abs_diff'] * 100:+.2f}pp",
                help=f"{m['conv_b']} conversions / {m['users_b']} users")
    col3.metric("Uplift", f"{m['uplift']:+.1f}%",
                help="Percentage change in conversion rate from A to B")
    col4.metric("p-value", f"{m['p_value']:.4f}",
                help="Probability of observing this result by chance if there is no real difference")

    # ── AI: Executive summary (auto-generated) ────────────────────────────────
    if openai_api_key:
        if "ai_exec_summary" not in st.session_state:
            with st.spinner("Generating executive summary..."):
                try:
                    st.session_state["ai_exec_summary"] = generate_exec_summary(
                        m, openai_api_key
                    )
                except Exception as e:
                    st.warning(f"Could not generate executive summary: {e}")
        if st.session_state.get("ai_exec_summary"):
            st.info(st.session_state["ai_exec_summary"])

    st.markdown("---")

    # ── Significance ──────────────────────────────────────────────────────────
    st.subheader("Statistical Significance")

    sig_col1, sig_col2, sig_col3 = st.columns(3)
    sig_col1.metric("Confidence Level", f"{min(m['confidence'], 99.9):.1f}%")
    sig_col2.metric("Significance Threshold", "95%")
    sig_col3.metric(
        "Result",
        "Significant ✓" if m["significant"] else "Not Significant ✗",
    )

    st.markdown("---")

    # ── Recommendation ────────────────────────────────────────────────────────
    st.subheader("Recommendation")
    rec_text, widget_type = generate_recommendation(m)

    if widget_type == "success":
        st.success(rec_text)
    elif widget_type == "error":
        st.error(rec_text)
    else:
        st.warning(rec_text)

    # ── AI: Narrative analysis ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("AI Analysis")

    if not openai_api_key:
        st.caption("Add your OpenAI API key in the sidebar to enable AI-powered analysis.")
    else:
        btn_label = (
            "Regenerate AI Analysis"
            if "ai_narrative" in st.session_state
            else "Generate AI Analysis"
        )
        if st.button(btn_label, key="btn_narrative"):
            with st.spinner("Generating analysis..."):
                try:
                    st.session_state["ai_narrative"] = generate_narrative(
                        m, rec_text, openai_api_key
                    )
                except Exception as e:
                    st.error(f"Could not generate analysis: {e}")

        if "ai_narrative" in st.session_state:
            st.markdown(st.session_state["ai_narrative"])

    # ── AI: Follow-up Q&A chat ────────────────────────────────────────────────
    if openai_api_key and "ai_narrative" in st.session_state:
        st.markdown("---")
        st.subheader("Ask Follow-up Questions")

        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.get("chat_history"):
            if st.button("Clear chat history", key="btn_clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()

        if prompt := st.chat_input("Ask a question about these results..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            history = st.session_state.get("chat_history", [])

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = answer_ab_question(
                            prompt, m, history, openai_api_key
                        )
                    except Exception as e:
                        answer = f"Sorry, I could not generate a response: {e}"
                st.markdown(answer)

            st.session_state["chat_history"] = history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]

else:
    st.info(
        "Upload a CSV or Excel file with columns: **variant**, **users**, **conversions**.\n\n"
        "Example:\n```\nvariant,users,conversions\nA,1000,120\nB,1000,145\n```"
    )
