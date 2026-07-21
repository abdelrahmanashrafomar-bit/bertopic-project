import streamlit as st
import numpy as np
import torch
import os
import streamlit.components.v1 as components

# Import modular backend from demo.py
from demo import predict, _load_artifacts, _centroids, _topic_keywords, _topic_sizes, _lookup

# ---------------------------------------------------------------------------
# Streamlit configuration & page styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CFPB Complaint Topic Analyzer",
    page_icon="🏦",
    layout="wide",
)

# Custom CSS for modern design and rounded cards
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .keyword-badge {
        display: inline-block;
        background-color: #312E81;
        color: #EEF2F6;
        padding: 5px 10px;
        margin: 4px;
        border-radius: 12px;
        font-weight: 500;
        font-size: 13px;
        border: 1px solid #4338CA;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Warm-up / Cache Model
# ---------------------------------------------------------------------------
@st.cache_resource
def bootstrap_application():
    """Load model & artifacts once and keep in memory across user sessions."""
    _load_artifacts()
    return True

# ---------------------------------------------------------------------------
# Sidebar / System Stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/Consumer_Financial_Protection_Bureau_logo.svg", width=180)
    st.markdown("---")
    st.markdown("### 🖥️ Hardware Diagnostics")
    
    device = "CUDA GPU" if torch.cuda.is_available() else "CPU Mode"
    device_color = "green" if torch.cuda.is_available() else "orange"
    st.markdown(f"**Compute Device:** :{device_color}[{device}]")
    
    if torch.cuda.is_available():
        st.markdown(f"**GPU Model:** `{torch.cuda.get_device_name(0)}`")
    st.markdown("---")
    st.markdown("### 📊 Dataset Parameters")
    st.markdown("**Corpus Size:** 35,993 documents")
    st.markdown("**Embeddings:** F2LLM-1.7B (2048-D)")

# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
st.title("🏦 CFPB Consumer Complaint Topic Analyzer")
st.markdown("Analyze financial consumer narratives using fine-tuned semantic embeddings and unsupervised topic extraction.")
st.markdown("---")

# Bootstrap the heavy model
with st.spinner("🚀 Loading 1.7B parameter LLM encoder into memory... (This takes about 60 seconds on start)"):
    bootstrap_application()

# Clickable example templates
st.markdown("### 💡 Click an example to test:")
col_ex1, col_ex2, col_ex3 = st.columns(3)

example_1 = "I am a victim of identity theft. Someone opened a credit card in my name without my authorization. Please remove this from my credit report."
example_2 = "I demand validation of this debt under the FDCPA. The debt collector is trying to collect an amount I do not owe and has failed to verify the contract."
example_3 = "My mortgage servicer incorrectly applied my payment. They put my funds in suspense and charged me a late fee instead of updating the billing ledger."

example_text = ""
if col_ex1.button("💳 Identity Theft Example"):
    example_text = example_1
if col_ex2.button("📞 Debt Collection Example"):
    example_text = example_2
if col_ex3.button("🏠 Mortgage Dispute Example"):
    example_text = example_3

# Text entry columns
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📝 Text Narrative")
    user_input = st.text_area(
        "Paste the complaint text below:",
        value=example_text,
        placeholder="Type customer complaint narrative here...",
        height=220
    )
    run_btn = st.button("Classify Complaint", type="primary", use_container_width=True)

with col_output:
    st.subheader("🔮 Model Inference")
    
    if run_btn or example_text:
        if not user_input.strip():
            st.warning("Please type or select a complaint template first.")
        else:
            with st.spinner("Generating 2048-D embedding & running centroid matching..."):
                result = predict(user_input)
                
            if result:
                label = result["label"]
                topic_id = result["topic_id"]
                score = result["score"]
                keywords = result["keywords"]
                size = result["size"]
                low_conf = result["low_confidence"]
                
                # Card 1: Prediction
                st.markdown(f"""
                <div class="metric-card">
                    <p style="margin:0;color:#94A3B8;font-size:12px;text-transform:uppercase;">Predicted Category</p>
                    <h2 style="margin:5px 0;color:#38BDF8;">{label}</h2>
                    <p style="margin:0;color:#64748B;font-size:13px;">Topic ID: <b>{topic_id}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Card 2: Similarity Score Gauge
                score_pct = score * 100
                bar_color = "green" if score >= 0.22 else ("orange" if score >= 0.16 else "red")
                
                st.markdown(f"**Embedding Similarity (Cosine):** `{score_pct:.1f}%`")
                if bar_color == "green":
                    st.progress(score, text="High Confidence Match")
                elif bar_color == "orange":
                    st.progress(score, text="Moderate Confidence Match")
                else:
                    st.progress(score, text="Low Confidence Match")
                    st.warning("⚠️ Low similarity detected. This complaint might be a unique outlier.")
                
                # Card 3: Associated documents
                st.markdown(f"""
                <div style="margin-top:20px;" class="metric-card">
                    <p style="margin:0;color:#94A3B8;font-size:12px;text-transform:uppercase;">Cluster Density</p>
                    <h3 style="margin:5px 0;color:#34D399;">{size:,} complaints</h3>
                    <p style="margin:0;color:#64748B;font-size:13px;">Total documents clustered into this category during training</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Keywords
                st.markdown("**Semantic Keywords:**")
                badges = "".join([f'<span class="keyword-badge">✓ {kw}</span>' for kw in keywords[:6]])
                st.markdown(badges, unsafe_allow_html=True)
                
            else:
                st.error("Input was empty after cleaning.")

# ---------------------------------------------------------------------------
# Interactive Plot Embedding
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Interactive Cluster Visualizations")
st.markdown("Explore the multi-dimensional structure of the complaints database. Hover over nodes to inspect document distribution.")

tab_dist, tab_hier = st.tabs(["Intertopic Distance Map", "Hierarchy Tree Diagram"])

with tab_dist:
    html_dist_path = "outputs/intertopic_distance_map.html"
    if os.path.exists(html_dist_path):
        with open(html_dist_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=650, scrolling=True)
    else:
        st.info("HTML map file missing. Run step 'visualize' to generate charts.")

with tab_hier:
    html_hier_path = "outputs/topic_hierarchy.html"
    if os.path.exists(html_hier_path):
        with open(html_hier_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=650, scrolling=True)
    else:
        st.info("HTML tree diagram file missing. Run step 'visualize' to generate charts.")
