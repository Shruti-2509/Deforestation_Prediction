import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import base64
import io
import json
import re
from typing import Any, Dict, Optional

load_dotenv()

# ─────────────────────────────────────────────────────────────
# OPENROUTER / CHATBOT HELPERS
# ─────────────────────────────────────────────────────────────
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

CHATBOT_SYSTEM_PROMPT = """
You are the Forest Guard Deforestation AI Assistant.
Help users with forest protection, deforestation monitoring, fire prevention,
illegal logging, reforestation, sustainable land use, and risk interpretation.
Keep answers practical, safe, and easy to understand.
When the user asks for a solution, give actionable steps.
""".strip()


def get_openrouter_api_key():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if key:
        return key
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    return ""


def get_openrouter_client():
    api_key = get_openrouter_api_key()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def call_openrouter_chat(messages, model=OPENROUTER_MODEL):
    client = get_openrouter_client()
    if client is None:
        return None, "OPENROUTER_API_KEY not found"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=500,
        )
        text = response.choices[0].message.content
        return (text.strip() if text else ""), None
    except Exception as e:
        return None, str(e)


def build_conversation_with_system(system_prompt, conversation, limit=12):
    trimmed = conversation[-limit:] if len(conversation) > limit else conversation
    return [{"role": "system", "content": system_prompt}] + trimmed


def fallback_general_response(prompt):
    prompt_lower = prompt.lower()

    if "high risk" in prompt_lower or ("high" in prompt_lower and "risk" in prompt_lower):
        return (
            "I can see this is a high risk situation, and that means the forest needs help fast. "
            "The trees are under serious pressure, so we should keep an eye on the area with patrols and satellite images. "
            "If people are cutting trees illegally, it is important to stop that right away and protect the boundary. "
            "A quick, calm response is better than waiting, because damage can spread fast. "
            "I would say focus on safety, stop the threats, and keep the forest covered."
        )
    elif "medium risk" in prompt_lower or ("medium" in prompt_lower and "risk" in prompt_lower):
        return (
            "Medium risk means this forest is not safe yet, but there is still a good chance to fix it. "
            "We should increase monitoring and ask nearby communities to help protect the land. "
            "Using rules and support can stop more trees from being cut down. "
            "It also helps to give farmers better options so they do not need to clear new forest. "
            "The key is to slow the harm and give the forest time to recover."
        )
    elif "low risk" in prompt_lower or ("low" in prompt_lower and "risk" in prompt_lower):
        return (
            "A low risk result is good news and means the forest is mostly healthy now. "
            "We should keep the protection going with regular checks and simple rules. "
            "Try to let people use nearby land without hurting the trees or nature. "
            "Keep looking out for fires or illegal activity so the area stays safe. "
            "If we do that, the forest can stay strong for a long time."
        )
    elif "fire" in prompt_lower or "smoke" in prompt_lower or "burn" in prompt_lower:
        return (
            "Fire is one of the fastest ways a forest can get damaged, so it needs careful attention. "
            "Watch the area closely, keep fire breaks clear, and make sure response teams are ready. "
            "Do not allow open burning when it is dry, because that can start a big blaze quickly. "
            "Simple actions like that can stop a small spark from becoming a major problem. "
            "This is the best way to keep trees, animals, and people safer."
        )
    elif "logging" in prompt_lower or "illegal" in prompt_lower or "chainsaw" in prompt_lower:
        return (
            "Illegal logging means people are taking trees without permission, and it hurts the forest. "
            "It helps to use patrols, cameras, or drones to see who is entering the area. "
            "If you notice suspicious activity, tell the right people right away. "
            "Protect the parts of the forest that should stay untouched, and make sure the rules are enforced. "
            "That way the forest can stay healthy and the damage will be less."
        )
    elif "deforestation" in prompt_lower or "deforest" in prompt_lower:
        return (
            "Start by scheduling satellite checks and field patrols on a regular cycle. "
            "When you find new tree loss, block access and deploy enforcement or community response fast. "
            "Use those alerts to stop activities before they spread and keep a clear record of each incident."
        )
    elif "forest" in prompt_lower or "forestation" in prompt_lower or "forest protection" in prompt_lower:
        return (
            "Set up protected zones with clear boundaries and local stewards. "
            "Keep those areas off-limits to logging and grazing, and support nearby communities with alternative incomes. "
            "Pair that with regular verification so you can respond quickly if protection breaks down."
        )
    elif "reforest" in prompt_lower or "reforestation" in prompt_lower:
        return (
            "Choose native species and plant them in the most degraded areas first. "
            "Protect seedlings with guards or fenced plots until they are established. "
            "Track their progress and keep watering or tending them during the first critical months."
        )
    elif "sustain" in prompt_lower or "sustainable" in prompt_lower or "sustainable land" in prompt_lower:
        return (
            "Switch to farming methods that keep trees and soil healthy, like agroforestry or crop rotation. "
            "Allocate forest buffer areas and limit clearing to already degraded zones. "
            "Support smallholders with training so they can earn income without expanding into new forest land."
        )
    else:
        return (
            "I can help you understand forest risk, fire safety, illegal logging, and how to protect trees. "
            "You can ask me about the app features, the risk predictor, or the image analysis tool."
        )


def predictor_system_prompt(current_risk):
    return f"""
You are a forest-protection assistant inside a deforestation risk predictor.
The current predicted risk is: {current_risk} Risk.
Give concise, practical, and situation-aware advice.
If the user asks for solutions or actions:
- High risk: prioritize urgent response, enforcement, and monitoring.
- Medium risk: prioritize prevention, patrols, outreach, and mitigation.
- Low risk: prioritize protection, maintenance, and long-term sustainability.
Keep the tone helpful and clear.
""".strip()


def fallback_predictor_response(current_risk, prompt):
    p_lower = prompt.lower()

    if "solution" in p_lower or "action" in p_lower or "do" in p_lower:
        if current_risk == "High":
            return (
                "Oh no, a **High Risk** prediction is definitely something we need to take seriously. I've put together an emergency plan for you:\n\n"
                "First, we should task high-resolution satellite monitoring to see what's happening on the ground. Then, it's crucial to coordinate with law enforcement and set up checkpoints. Finally, we should look into freezing land titles to prevent any further damage."
            )
        elif current_risk == "Medium":
            return (
                "It looks like this area is in a bit of a tricky spot. A **Medium Risk** means we still have time to turn things around! \n\n"
                "I'd suggest we start by increasing patrol frequency immediately. It's also a great time to start community outreach and maybe offer some agroforestry subsidies to help the locals protect the land."
            )
        else:
            return (
                "That's wonderful news! A **Low Risk** level means the ecosystem is currently stable. \n\n"
                "To keep it that way, I recommend designating the area as a permanent reserve. We can also support the local community with sustainable harvesting practices and keep up with long-term biodiversity research."
            )
    else:
        return (
            f"I'm here to help you navigate this **{current_risk} Risk** result. Feel free to ask me for 'solutions' or what 'actions' we should take next! I'm all ears."
        )


# ─────────────────────────────────────────────────────────────
# AGENTIC AI LAYER

def detect_user_intent(prompt: str):
    p = prompt.lower()

    if any(word in p for word in ["fire", "smoke", "burn", "wildfire"]):
        return "fire_prevention"

    if any(word in p for word in ["illegal logging", "logging", "chainsaw", "cut trees"]):
        return "illegal_logging"

    if any(word in p for word in ["reforest", "reforestation", "plant trees"]):
        return "reforestation"

    if any(word in p for word in ["sustain", "sustainable", "agroforestry", "land use"]):
        return "sustainable_land"

    if any(word in p for word in ["deforest", "forest loss", "tree loss"]):
        return "deforestation"

    if any(word in p for word in ["risk", "high risk", "medium risk", "low risk"]):
        return "risk_explanation"

    if any(word in p for word in ["satellite", "image", "scan", "detect"]):
        return "satellite_monitoring"

    return "general"


def fallback_chatbot_response(prompt: str):
    intent = detect_user_intent(prompt)

    if intent == "fire_prevention":
        return (
            "Forest fire prevention steps:\n"
            "1. Clear dry leaves and dead branches.\n"
            "2. Create firebreaks around risky zones.\n"
            "3. Stop open burning in dry weather.\n"
            "4. Keep patrols active during hot months.\n"
            "5. Use early smoke detection where possible."
        )

    elif intent == "illegal_logging":
        return (
            "Illegal logging can be reduced by:\n"
            "1. Increasing patrols near forest borders.\n"
            "2. Using cameras, drones, or satellite checks.\n"
            "3. Blocking unauthorized vehicle access.\n"
            "4. Reporting suspicious activity quickly.\n"
            "5. Working with local communities for protection."
        )

    elif intent == "reforestation":
        return (
            "For reforestation:\n"
            "1. Plant native tree species.\n"
            "2. Start with the most damaged land.\n"
            "3. Protect young plants from grazing and fire.\n"
            "4. Water and monitor them during early growth.\n"
            "5. Track survival rate over time."
        )

    elif intent == "sustainable_land":
        return (
            "Sustainable land use includes:\n"
            "1. Agroforestry\n"
            "2. Crop rotation\n"
            "3. Controlled grazing\n"
            "4. Soil conservation\n"
            "5. Limiting clearing to already degraded areas"
        )

    elif intent == "deforestation":
        return (
            "To reduce deforestation:\n"
            "1. Monitor forest area regularly.\n"
            "2. Stop illegal land clearing quickly.\n"
            "3. Support local people with sustainable jobs.\n"
            "4. Restore damaged zones.\n"
            "5. Use alerts to react early."
        )

    elif intent == "risk_explanation":
        return (
            "Risk level meaning:\n"
            "- Low risk: forest is mostly safe.\n"
            "- Medium risk: forest needs more monitoring.\n"
            "- High risk: immediate action is needed."
        )

    elif intent == "satellite_monitoring":
        return (
            "Satellite monitoring helps detect forest change early.\n"
            "It can show tree loss, fire damage, cloud cover, and land clearing.\n"
            "Regular satellite checks are useful for fast response."
        )

    else:
        return (
            "I can help with deforestation, forest protection, fire prevention, illegal logging, "
            "reforestation, sustainable land use, and satellite monitoring."
        )


def get_chatbot_response(prompt: str):
    api_key = get_openrouter_api_key()

    if api_key:
        messages = [
            {"role": "system", "content": CHATBOT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        response_text, err = call_openrouter_chat(messages)
        if response_text:
            return response_text
        return fallback_chatbot_response(prompt)

    return fallback_chatbot_response(prompt)
# ─────────────────────────────────────────────────────────────

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Safely extract JSON from model output."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return {}
    return {}


def simple_image_scan(image):
    """
    Lightweight image-analysis tool for agentic responses.
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    red_mean = float(arr[:, :, 0].mean())
    green_mean = float(arr[:, :, 1].mean())
    blue_mean = float(arr[:, :, 2].mean())

    if green_mean > red_mean + 0.05:
        risk = "Low"
        summary = "The image looks vegetation-heavy and healthier."
    elif red_mean > green_mean + 0.03:
        risk = "High"
        summary = "The image looks more barren or disturbed."
    else:
        risk = "Medium"
        summary = "The image shows a mixed pattern."

    return {
        "risk": risk,
        "summary": summary,
        "red_mean": red_mean,
        "green_mean": green_mean,
        "blue_mean": blue_mean,
    }


def create_action_plan(risk_level: str, user_prompt: str) -> str:
    risk_level = (risk_level or "medium").lower()

    if risk_level == "high":
        return (
            "Urgent action plan:\n"
            "1. Send an immediate field alert.\n"
            "2. Increase satellite and drone monitoring.\n"
            "3. Inform local forest officers.\n"
            "4. Check for illegal logging or fire risk.\n"
            "5. Block further land damage and start enforcement."
        )
    elif risk_level == "medium":
        return (
            "Preventive action plan:\n"
            "1. Increase patrol frequency.\n"
            "2. Monitor the area daily or weekly.\n"
            "3. Talk to local communities about protection.\n"
            "4. Watch for fires, clearing, or land misuse.\n"
            "5. Apply mitigation before the problem grows."
        )
    else:
        return (
            "Maintenance action plan:\n"
            "1. Keep regular monitoring active.\n"
            "2. Protect the healthy forest zone.\n"
            "3. Support sustainable land use.\n"
            "4. Track small changes early.\n"
            "5. Continue reforestation and conservation."
        )


class ForestGuardAgent:
    def __init__(self):
        self.system_prompt = """
You are ForestGuardAgent, an agentic AI for forest protection.

Your job:
1. Understand the user's goal.
2. Decide whether you need a tool.
3. Use the best tool.
4. Give a practical final answer.

Return ONLY JSON in this format:
{
  "tool": "chat|risk_summary|image_scan|action_plan|alert",
  "reason": "short reason",
  "priority": "low|medium|high",
  "final_answer": "final response to the user"
}
""".strip()

    def _build_messages(self, user_prompt: str, current_risk: Optional[str] = None, has_image: bool = False):
        context = f"""
Current risk: {current_risk if current_risk else "Unknown"}
Image available: {has_image}
User request: {user_prompt}
""".strip()

        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Choose the best tool and respond only in JSON.\n\n"
                    f"{context}"
                )
            }
        ]

    def decide(self, user_prompt: str, current_risk: Optional[str] = None, has_image: bool = False):
        messages = self._build_messages(user_prompt, current_risk=current_risk, has_image=has_image)
        raw_response, err = call_openrouter_chat(messages)

        if not raw_response:
            lower = user_prompt.lower()
            if has_image:
                return {"tool": "image_scan", "reason": "User uploaded an image", "priority": "medium", "final_answer": ""}
            if "solution" in lower or "action" in lower or "what should i do" in lower:
                return {"tool": "action_plan", "reason": "User asked for actions", "priority": "medium", "final_answer": ""}
            if "risk" in lower or "status" in lower:
                return {"tool": "risk_summary", "reason": "User asked for risk info", "priority": "medium", "final_answer": ""}
            return {"tool": "chat", "reason": "General question", "priority": "low", "final_answer": fallback_general_response(user_prompt)}

        data = safe_json_loads(raw_response)

        if not data:
            return {
                "tool": "chat",
                "reason": "Model output was not valid JSON",
                "priority": "low",
                "final_answer": raw_response.strip(),
            }

        if "tool" not in data:
            data["tool"] = "chat"
        if "priority" not in data:
            data["priority"] = "medium"
        if "final_answer" not in data:
            data["final_answer"] = raw_response.strip()

        return data

    def run(self, user_prompt: str, df=None, current_risk: Optional[str] = None, image=None) -> str:
        decision = self.decide(
            user_prompt=user_prompt,
            current_risk=current_risk,
            has_image=image is not None
        )

        tool = decision.get("tool", "chat")

        if tool == "image_scan" and image is not None:
            result = simple_image_scan(image)
            action_plan = create_action_plan(result["risk"], user_prompt)
            return (
                f"Agent action: image scanned.\n\n"
                f"Detected risk: {result['risk']}\n"
                f"Image summary: {result['summary']}\n\n"
                f"Next action:\n{action_plan}"
            )

        if tool == "risk_summary" and df is not None:
            avg_rate = df["deforestation_rate"].mean()
            high_count = len(df[df["label"] == "High"])
            low_count = len(df[df["label"] == "Low"])
            med_count = len(df[df["label"] == "Medium"])

            return (
                "Agent action: dataset reviewed.\n\n"
                f"Average deforestation rate: {avg_rate:.2f}%\n"
                f"Low risk areas: {low_count}\n"
                f"Medium risk areas: {med_count}\n"
                f"High risk areas: {high_count}\n\n"
                f"Answer: {decision.get('final_answer', '')}"
            )

        if tool == "action_plan":
            risk_to_use = current_risk if current_risk else "Medium"
            plan = create_action_plan(risk_to_use, user_prompt)
            return (
                f"Action: created an action plan for {risk_to_use} risk.\n\n"
                f"{plan}\n\n"
                f"\n{decision.get('final_answer', '')}"
            )

        if tool == "alert":
            return (
                "Agent action: alert mode activated.\n\n"
                "Please check the area immediately and notify forest authorities."
            )

        return decision.get("final_answer", fallback_general_response(user_prompt))


# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deforestation Predictor",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS FOR PREMIUM AESTHETICS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #f0f4f8;
        color: #1a202c;
    }

    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: #059669;
    }

    /* Headers */
    h1, h2, h3 {
        color: #10b981 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }

    /* Premium White Cards */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }

    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.95;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
        transform: scale(1.02);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e2e8f0;
        border-radius: 10px 10px 0px 0px;
        padding: 12px 24px;
        border: 1px solid #cbd5e0;
        border-bottom: none;
        color: #4a5568;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: #10b981 !important;
        border-top: 3px solid #10b981;
        font-weight: 700;
    }

    /* Prediction Result Cards */
    .pred-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 6px solid #10b981;
        padding: 24px; border-radius: 12px; color: #064e3b; font-size: 26px; text-align: center; font-weight: 800;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);
    }
    .pred-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 6px solid #f59e0b;
        padding: 24px; border-radius: 12px; color: #78350f; font-size: 26px; text-align: center; font-weight: 800;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.1);
    }
    .pred-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #ef4444;
        padding: 24px; border-radius: 12px; color: #7f1d1d; font-size: 26px; text-align: center; font-weight: 800;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA & MODEL LOADING (CACHED)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/data.csv')
        # Handle Missing values
        df['incremento'] = df['incremento'].fillna(df['incremento'].median())

        # Derived features
        df['deforestation_rate'] = df['desmatado'] / df['area'] * 100
        df['forest_cover_pct']   = df['floresta']  / df['area'] * 100
        df['cloud_pct']          = df['nuvem']     / df['area'] * 100

        # Target Label
        def classify_deforestation(rate):
            if rate < 5: return 'Low'
            elif rate < 20: return 'Medium'
            else: return 'High'

        df['label'] = df['deforestation_rate'].apply(classify_deforestation)
        return df
    except Exception:
        return None


@st.cache_resource
def train_model(df):
    features = ['area','desmatado','incremento','floresta','nuvem',
                'nao_observado','nao_floresta','hidrografia',
                'deforestation_rate','forest_cover_pct','cloud_pct']

    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_scaled, y)

    return rf, scaler, le, features


df = load_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌳 Forest Guard")
    st.markdown("*AI-Powered Analysis Tool*")
    st.markdown("---")
    page = st.radio("Navigation", ["Overview & Analytics", "Tabular Predictor", "Image Classification", "Chatbot Assistant"])
    st.markdown("---")


# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW & ANALYTICS
# ─────────────────────────────────────────────────────────────
if page == "Overview & Analytics":
    st.title("🌍 Deforestation Analytics Dashboard")
    st.markdown("Discover insights and trends from historical satellite and tabular data.")

    if df is not None:
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{len(df):,}")
        m2.metric("Avg Deforestation Rate", f"{df['deforestation_rate'].mean():.2f}%")
        m3.metric("High Risk Areas", f"{len(df[df['label']=='High']):,}")
        m4.metric("Average Forest Cover", f"{df['forest_cover_pct'].mean():.2f}%")

        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Deforestation Trend Over Time")
            yearly = df.groupby('ano')['desmatado'].sum().reset_index()
            fig1 = px.area(yearly, x='ano', y='desmatado',
                           color_discrete_sequence=['#ef4444'],
                           template='plotly_white')
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Forest vs Deforested Scatter")
            sample_df = df.sample(min(1000, len(df)))
            fig2 = px.scatter(sample_df, x='floresta', y='desmatado', color='label',
                              color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'},
                              template='plotly_white')
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("Risk Distribution")
            fig3 = px.pie(df, names='label', hole=0.6,
                          color='label',
                          color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'},
                          template='plotly_white')
            fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.error("Data could not be loaded. Please ensure 'data/data.csv' exists.")

# ─────────────────────────────────────────────────────────────
# PAGE: TABULAR PREDICTOR
# ─────────────────────────────────────────────────────────────
elif page == "Tabular Predictor":
    st.title("🔮 Interactive Tabular Predictor")
    st.markdown("Adjust the parameters below to predict the deforestation risk in real-time.")

    if df is not None:
        rf_model, scaler, le, features = train_model(df)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            area = st.slider("Total Area (km²)", 100, 200000, 5000)
            desmatado = st.slider("Deforested Area (km²)", 0, int(area), int(area*0.1))
            floresta = st.slider("Forest Area (km²)", 0, int(area), int(area*0.8))
            incremento = st.slider("Increment (km²)", 0.0, 5000.0, 100.0)

        with col2:
            nuvem = st.slider("Cloud Cover (km²)", 0, int(area), 0)
            nao_observado = st.number_input("Not Observed (km²)", 0, int(area), 0)
            nao_floresta = st.number_input("Non-Forest Area (km²)", 0, int(area), 0)
            hidrografia = st.number_input("Hydrography (km²)", 0, int(area), 0)

        st.markdown("</div>", unsafe_allow_html=True)

        # Calculate derived features
        deforestation_rate = (desmatado / area * 100) if area > 0 else 0
        forest_cover_pct = (floresta / area * 100) if area > 0 else 0
        cloud_pct = (nuvem / area * 100) if area > 0 else 0

        if st.button("🚀 Predict Risk Level", use_container_width=True):
            with st.spinner("Analyzing parameters..."):
                time.sleep(0.5)

                input_data = np.array([[area, desmatado, incremento, floresta, nuvem,
                                        nao_observado, nao_floresta, hidrografia,
                                        deforestation_rate, forest_cover_pct, cloud_pct]])

                input_scaled = scaler.transform(input_data)
                pred_idx = rf_model.predict(input_scaled)[0]
                pred_prob = rf_model.predict_proba(input_scaled)[0]
                prediction = le.inverse_transform([pred_idx])[0]

                st.session_state.last_prediction = prediction
                st.session_state.predictor_messages = [
                    {"role": "assistant", "content": f"The area is currently at **{prediction} Risk**. Ask me for 'solutions' or 'actions' to address this!"}
                ]

                st.markdown("### Prediction Result")
                if prediction == 'Low':
                    st.markdown("<div class='pred-low'>🟢 LOW RISK</div>", unsafe_allow_html=True)
                elif prediction == 'Medium':
                    st.markdown("<div class='pred-medium'>🟡 MEDIUM RISK</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='pred-high'>🔴 HIGH RISK</div>", unsafe_allow_html=True)

                fig = go.Figure(data=[go.Bar(
                    x=le.classes_,
                    y=pred_prob,
                    marker_color=['#ef4444' if c=='High' else '#f59e0b' if c=='Medium' else '#10b981' for c in le.classes_]
                )])
                fig.update_layout(title="Confidence Scores", template='plotly_white', height=300,
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        # Integrated Predictor Agent Assistant
        if "last_prediction" in st.session_state:
            st.markdown("---")
            st.subheader(f"🤖 Solutions Assistant for {st.session_state.last_prediction} Risk")

            if "predictor_messages" not in st.session_state:
                st.session_state.predictor_messages = []

            for msg in st.session_state.predictor_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if p_prompt := st.chat_input("Ask for solutions...", key="predictor_chat"):
                st.chat_message("user").markdown(p_prompt)
                st.session_state.predictor_messages.append({"role": "user", "content": p_prompt})

                agent = ForestGuardAgent()
                p_res = agent.run(
                    user_prompt=p_prompt,
                    df=df,
                    current_risk=st.session_state.last_prediction,
                    image=None
                )

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in p_res.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(p_res)
                st.session_state.predictor_messages.append({"role": "assistant", "content": p_res})

# ─────────────────────────────────────────────────────────────
# PAGE: IMAGE CLASSIFICATION
# ─────────────────────────────────────────────────────────────
elif page == "Image Classification":
    st.title("🛰️ Satellite Image Classification")
    st.markdown("Upload a satellite image to detect signs of deforestation.")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Analysis & Classification")
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Scanning for boundaries and deforestation patterns..."):
                    time.sleep(1)

                    draw_img = image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    width, height = image.size
                    grid_size = 12
                    cell_w, cell_h = width // grid_size, height // grid_size

                    forest_pts = []
                    deforest_pts = []

                    for i in range(grid_size):
                        for j in range(grid_size):
                            left, top = j * cell_w, i * cell_h
                            right, bottom = (j + 1) * cell_w, (i + 1) * cell_h

                            cell = image.crop((left, top, right, bottom))
                            arr = np.array(cell, dtype=np.float32) / 255.0
                            g_mean = arr[:,:,1].mean()
                            r_mean = arr[:,:,0].mean()

                            if (g_mean > r_mean) and (g_mean > 0.2):
                                forest_pts.append((left, top, right, bottom))
                            else:
                                deforest_pts.append((left, top, right, bottom))

                    if forest_pts:
                        f_min_x = min(p[0] for p in forest_pts)
                        f_min_y = min(p[1] for p in forest_pts)
                        f_max_x = max(p[2] for p in forest_pts)
                        f_max_y = max(p[3] for p in forest_pts)
                        draw.rectangle([f_min_x, f_min_y, f_max_x, f_max_y], outline=(16, 185, 129), width=6)
                        draw.rectangle([f_min_x, f_min_y, f_min_x+100, f_min_y+25], fill=(16, 185, 129))
                        draw.text((f_min_x+5, f_min_y+5), "FOREST", fill=(255, 255, 255))

                    if deforest_pts:
                        d_min_x = min(p[0] for p in deforest_pts)
                        d_min_y = min(p[1] for p in deforest_pts)
                        d_max_x = max(p[2] for p in deforest_pts)
                        d_max_y = max(p[3] for p in deforest_pts)
                        draw.rectangle([d_min_x, d_min_y, d_max_x, d_max_y], outline=(239, 68, 68), width=6)
                        draw.rectangle([d_min_x, d_min_y, d_min_x+130, d_min_y+25], fill=(239, 68, 68))
                        draw.text((d_min_x+5, d_min_y+5), "DEFORESTED", fill=(255, 255, 255))

                    deforestation_pct = (len(deforest_pts) / (grid_size**2)) * 100

                    st.markdown("### Analysis Result")
                    st.image(draw_img, caption="Green: Forest Zone | Red: Deforested Zone", use_column_width=True)

                    if deforestation_pct < 20:
                        st.markdown("<div class='pred-low'>🌲 FOREST (Healthy)</div>", unsafe_allow_html=True)
                        st.success(f"Minimal deforestation detected ({deforestation_pct:.1f}% affected area).")
                    elif deforestation_pct < 50:
                        st.markdown("<div class='pred-medium'>🟡 FRAGMENTED FOREST</div>", unsafe_allow_html=True)
                        st.warning(f"Moderate deforestation detected ({deforestation_pct:.1f}% affected area).")
                    else:
                        st.markdown("<div class='pred-high'>🪓 HIGHLY DEFORESTED</div>", unsafe_allow_html=True)
                        st.error(f"Significant deforestation detected ({deforestation_pct:.1f}% affected area)!")

                    st.markdown("#### Color Channel Distribution (Overall)")
                    arr_full = np.array(image, dtype=np.float32) / 255.0
                    st.progress(min(float(arr_full[:,:,0].mean()), 1.0), text="Red (Barren/Soil)")
                    st.progress(min(float(arr_full[:,:,1].mean()), 1.0), text="Green (Vegetation)")
                    st.progress(min(float(arr_full[:,:,2].mean()), 1.0), text="Blue (Water/Shadows)")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: CHATBOT ASSISTANT
# ─────────────────────────────────────────────────────────────
elif page == "Chatbot Assistant":
    st.title("🤖 Deforestation AI Assistant")
    st.markdown(
        "Ask me about deforestation, forest protection, fire prevention, illegal logging, "
        "reforestation, sustainable land use, or satellite monitoring."
    )

    api_key = get_openrouter_api_key()
    if api_key:
        st.success(f"OpenRouter connected. Model: `{OPENROUTER_MODEL}`")
    else:
        st.warning("No OpenRouter API key found. The app will use the built-in fallback chatbot.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 🌳 I'm your **Forest Guard AI Assistant**.\n\n"
                    
                )
            }
        ]

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! 🌳 How can I help you today?"
            }
        ]
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "Ask a question about forests, deforestation, fire, logging, or reforestation..."
    ):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response_text = get_chatbot_response(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.03)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
