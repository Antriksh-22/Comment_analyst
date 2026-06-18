from dotenv import load_dotenv
load_dotenv()
import plotly.express as px
import streamlit as st
import torch
import pickle
import google.generativeai as genai
import streamlit as st
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# GEMINI CONFIG
# =====================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "./final_intent_model"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "./final_intent_model"
    )

    with open("label_encoder.pkl","rb") as f:
        label_encoder = pickle.load(f)

    return tokenizer, model, label_encoder


tokenizer, model, label_encoder = load_model()

# =====================================================
# RULE OVERRIDES
# =====================================================

def rule_based_override(text):

    text = text.lower()

    if (
        "price" in text or
        "cost" in text or
        "how much" in text or
        "rudraksha" in text or
        "buy" in text
    ):
        return "product_inquiry"

    if (
        "thank you" in text or
        "thanks" in text or
        "great product" in text or
        "awesome" in text
    ):
        return "feedback"

    return None


# =====================================================
# INTENT
# =====================================================

def predict_intent(text):

    rule = rule_based_override(text)

    if rule:
        return rule, 1.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs.pop("token_type_ids", None)

    with torch.no_grad():

        outputs = model(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    confidence = float(
        probs.max()
    )

    pred = torch.argmax(
        probs,
        dim=1
    ).item()

    intent = label_encoder.inverse_transform(
        [pred]
    )[0]

    return intent, confidence


# =====================================================
# PRIORITY
# =====================================================

def get_priority(text):

    text = text.lower()

    keywords = [

        "refund",
        "fraud",
        "scam",
        "fake",
        "money back",
        "not received",
        "urgent",
        "broken",
        "damaged"

    ]

    for k in keywords:

        if k in text:
            return "HIGH"

    return "LOW"


# =====================================================
# ROUTING
# =====================================================

def get_team(intent):

    if intent == "refund_return":
        return "complaint_team"

    if intent == "product_inquiry":
        return "sales_team"

    if intent == "account_issue":
        return "customer_support"

    if intent == "shipping_delivery":
        return "customer_support"

    return "customer_support"


# =====================================================
# GEMINI REPLY
# =====================================================

def generate_reply(comment, intent):

    try:

        prompt = f"""
Customer Comment:
{comment}

Intent:
{intent}

Generate a professional customer support reply.
Maximum 50 words.
"""

        response = gemini_model.generate_content(
            prompt
        )

        return response.text

    except:

        return "Thank you for contacting us. Please send us a DM and our team will assist you."


# =====================================================
# PIPELINE
# =====================================================

def process_comment(comment):

    intent, confidence = predict_intent(
        comment
    )

    priority = get_priority(
        comment
    )

    team = get_team(
        intent
    )

    reply = generate_reply(
        comment,
        intent
    )

    return {

        "intent": intent,
        "confidence": round(
            confidence,
            4
        ),
        "priority": priority,
        "team": team,
        "reply": reply

    }


# =====================================================
# UI
# =====================================================

st.title(
    "Instagram & Facebook Comment Manager"
)

comment = st.text_area(
    "Enter Comment"
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if st.button(
    "Analyze"
):

    if comment.strip():

        result = process_comment(
            comment
        )

        st.success(
            "Analysis Complete"
        )

        st.write(
            "### Intent"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Intent",
            result["intent"]
            )

        col2.metric(
            "Confidence",
            f"{result['confidence']*100:.2f}%"
         )

        col3.metric(
            "Priority",
            result["priority"]
            )

        col4.metric(
            "Team",
            result["team"]
            )

        st.write(
            "### AI Reply"
        )

        st.write(
            result["reply"]
        )

# =====================================================
# BULK CSV ANALYSIS
# =====================================================

if uploaded_file is not None:

    import pandas as pd

    df = pd.read_csv(uploaded_file)

    results = []

    column_name = df.columns[0]

    for comment in df[column_name]:

        result = process_comment(comment)

        results.append({

            "comment": comment,
            "intent": result["intent"],
            "priority": result["priority"],
            "team": result["team"]

        })

    result_df = pd.DataFrame(results)

    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.subheader("Results")

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.divider()

    st.header("Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Comments",
        len(result_df)
    )

    col2.metric(
        "Refund Cases",
        len(
            result_df[
                result_df["intent"] == "refund_return"
            ]
        )
    )

    col3.metric(
        "Sales Leads",
        len(
            result_df[
                result_df["intent"] == "product_inquiry"
            ]
        )
    )

    col4.metric(
        "High Priority",
        len(
            result_df[
                result_df["priority"] == "HIGH"
            ]
        )
    )

    # =====================================================
    # INTENT CHART
    # =====================================================

    intent_counts = (
        result_df["intent"]
        .value_counts()
        .reset_index()
    )

    intent_counts.columns = [
        "Intent",
        "Count"
    ]

    fig1 = px.bar(
        intent_counts,
        x="Intent",
        y="Count",
        title="Intent Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================================
    # PRIORITY CHART
    # =====================================================

    priority_counts = (
        result_df["priority"]
        .value_counts()
        .reset_index()
    )

    priority_counts.columns = [
        "Priority",
        "Count"
    ]

    fig2 = px.pie(
        priority_counts,
        names="Priority",
        values="Count",
        title="Priority Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =====================================================
    # TEAM CHART
    # =====================================================

    team_counts = (
        result_df["team"]
        .value_counts()
        .reset_index()
    )

    team_counts.columns = [
        "Team",
        "Count"
    ]

    fig3 = px.bar(
        team_counts,
        x="Team",
        y="Count",
        title="Team Routing Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

    csv = result_df.to_csv(index=False)

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="processed_comments.csv",
        mime="text/csv"
    )