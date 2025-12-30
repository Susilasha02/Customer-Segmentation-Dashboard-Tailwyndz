import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
import plotly.graph_objects as go

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Segmentation SaaS",
    layout="wide"
)


REQUIRED_COLUMNS = [
    "customer_id",
    "days_since_last_purchase",
    "total_transactions",
    "total_sales",
    "avg_purchase_value",
    "avg_items_per_transaction",
    "age",
    "membership_years",
    "avg_discount_used",
    "distance_to_store"
]

MAX_ROWS = 100_000


# LOAD ARTIFACTS

@st.cache_resource
def load_artifacts():
    return {
        "features": joblib.load("feature_columns.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "pca": joblib.load("pca.pkl"),
        "kmeans": joblib.load("kmeans.pkl"),
        "birch": joblib.load("birch.pkl"),
        "gmm": joblib.load("gmm.pkl")
    }

artifacts = load_artifacts()


# =========================================================
# GLOBAL STYLES
# =========================================================
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #F3EEFF;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6A11CB, #8B5CF6);
    color: white;
}

section[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600;
}

/* Hero */
.hero {
    background: linear-gradient(90deg, #6A11CB, #B721FF);
    padding: 52px;
    border-radius: 30px;
    color: white;
    text-align: center;
    margin-bottom: 36px;
}

/* Upload section */
.upload-box {
    background: #EEE9FF;
    padding: 26px;
    border-radius: 20px;
    margin-bottom: 36px;
}

/* KPI card */
.kpi-card {
    background: white;
    border-radius: 22px;
    padding: 26px;
    border: 1px solid #DDD6FE;
    box-shadow: 0 12px 28px rgba(0,0,0,0.08);
}

.kpi-icon {
    font-size: 28px;
}

.kpi-label {
    font-size: 15px;
    font-weight: 600;
    color: #6B7280;
}

.kpi-value {
    font-size: 34px;
    font-weight: 900;
    color: #4C1D95;
    margin-top: 6px;
}

.card {
    background: #FFFFFF;
    border-radius: 22px;
    padding: 24px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 14px 28px rgba(0,0,0,0.08);
}
.card-title {
    font-size: 18px;
    font-weight: 800;
    color: #4C1D95;
    margin-bottom: 12px;
}


</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("## Navigation")

page = st.sidebar.radio(
    "",
    ["Home", "Model Analysis", "Customer Analysis", "Action Board"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Customer Segmentation SaaS")



# =========================================================
# HOME PAGE
# =========================================================
if page == "Home":
   
    st.markdown("""
    <div class="hero">
        <h1>Customer Segmentation Analysis</h1>
        <p style="font-size:18px;">
        Analyze customer value, behavior, and engagement to drive smarter
        marketing, retention, and revenue decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("""
    <div class="upload-box">
        <h3>Upload Customer Dataset</h3>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()

        # --------------------------------------------------
        # COLUMN VALIDATION & MAPPING
        # --------------------------------------------------
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing_cols:
            st.warning("Required columns missing. Please map your columns.")
            column_mapping = {}

            for col in missing_cols:
                column_mapping[col] = st.selectbox(
                    f"Map column for '{col}'",
                    options=["-- Select --"] + list(df.columns),
                    key=f"map_{col}"
                )

            if st.button("Apply Mapping"):
                if "-- Select --" in column_mapping.values():
                    st.error("Please map all required columns.")
                    st.stop()

                for req, user_col in column_mapping.items():
                    df[req] = df[user_col]

        # --------------------------------------------------
        # SAMPLING (PERFORMANCE SAFETY)
        # --------------------------------------------------
        if len(df) > MAX_ROWS:
            df = df.sample(MAX_ROWS, random_state=42)
            st.info("Dataset sampled to 100,000 rows for performance")

        # --------------------------------------------------
        # DERIVE RFM BASE VARIABLES
        # --------------------------------------------------
        df["recency"] = df["days_since_last_purchase"]
        df["frequency"] = df["total_transactions"]
        df["monetary"] = df["total_sales"]

        # --------------------------------------------------
        # STORE CLEAN DATA FOR OTHER PAGES
        # --------------------------------------------------
        st.session_state["data"] = df
        st.success("Dataset loaded successfully")

   

        # KPI CARDS
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">👥</div>
                <div class="kpi-label">Total Customers</div>
                <div class="kpi-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">₹</div>
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value">₹{df['monetary'].sum():,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🛒</div>
                <div class="kpi-label">Avg Purchase Value</div>
                <div class="kpi-value">₹{df['avg_purchase_value'].mean():,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">⏱</div>
                <div class="kpi-label">Avg Recency</div>
                <div class="kpi-value">{df['recency'].mean():.0f} days</div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# MODEL ANALYSIS PAGE
# =========================================================
elif page == "Model Analysis":
    if "data" not in st.session_state:
        st.warning("Please upload a dataset on the Home page first.")
        st.stop()

    data = st.session_state["data"].copy()

    # Rebuild feature pipeline safely
    FEATURE_COLUMNS = artifacts["features"]
    X = data[FEATURE_COLUMNS].copy()
    X_scaled = artifacts["scaler"].transform(X)
    X_pca = artifacts["pca"].transform(X_scaled)

    st.markdown("## Model Analysis")

    # -----------------------------------------------------
    # MODEL SELECTION PANEL
    # -----------------------------------------------------
    st.markdown("### Select Clustering Model")

    model_choice = st.radio(
        "",
        ["KMeans", "Birch", "Gaussian Mixture Model (GMM)"],
        horizontal=True
    )

    if model_choice == "KMeans":
        model = artifacts["kmeans"]
        clusters = model.predict(X_pca)
        model_desc = "KMeans forms compact, spherical clusters by minimizing within-cluster variance."
    elif model_choice == "Birch":
        model = artifacts["birch"]
        clusters = model.predict(X_pca)
        model_desc = "BIRCH incrementally builds clusters, efficient for large datasets."
    else:
        model = artifacts["gmm"]
        clusters = model.predict(X_pca)
        model_desc = "GMM models probabilistic clusters allowing overlapping boundaries."

    st.caption(model_desc)
    st.markdown("""
<div style="
    background:#F5F3FF;
    padding:14px 18px;
    border-radius:14px;
    margin-top:10px;
    border-left:6px solid #8B5CF6;
    font-size:14px;
">
<b> Model Insight </b><br>
• <b>KMeans</b> → Best for clearly separated customer groups<br>
• <b>Birch</b> → Efficient for large datasets<br>
• <b>GMM</b> → Best when customer behavior overlaps
</div>
""", unsafe_allow_html=True)


    data["cluster"] = clusters

    # -----------------------------------------------------
    # CLUSTER PROFILE TABLE
    # -----------------------------------------------------

    st.markdown("### Cluster Profile")
    profile = (
        data.groupby("cluster")[[
            "frequency",
            "recency",
            "monetary",
            "avg_purchase_value",
            "avg_items_per_transaction"
        ]]
        .mean()
        .round(2)
        .reset_index()
    )

    
    styled_profile = (
    profile.style
    .format("{:.2f}", subset=[
        "frequency", "recency", "monetary",
        "avg_purchase_value", "avg_items_per_transaction"
    ])
    .set_properties(**{
        "border": "1px solid #E5E7EB",
        "font-size": "14px",
        "text-align": "center"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("font-weight", "800"),
            ("font-size", "14px"),
            ("background-color", "#F5F3FF"),
            ("color", "#4C1D95"),
            ("border", "1px solid #DDD6FE")
        ]},
        {"selector": "tr:nth-child(even)", "props": [
            ("background-color", "#FAF5FF")
        ]},
        {"selector": "tr:nth-child(odd)", "props": [
            ("background-color", "#FFFFFF")
        ]},
        {"selector": "td", "props": [
            ("padding", "10px")
        ]}
    ])
)

    st.write(styled_profile)



   
    # -----------------------------------------------------
    # PCA + MODEL QUALITY (ALIGNED LAYOUT)
    # -----------------------------------------------------
    st.markdown("### Customer Clusters (PCA Projection)")

    pca_col, metric_col = st.columns([3.2, 1.5], gap="large")

    # ---------- PCA PLOT CARD ----------
    with pca_col:
        st.markdown("""
        <div style="
            background:white;
            padding:18px;
            border-radius:18px;
            border:1px solid #DDD6FE;
            box-shadow:0 10px 24px rgba(0,0,0,0.08);
        ">
        """, unsafe_allow_html=True)

        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=data["cluster"].astype(str),
            labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
            opacity=0.65
        )

        
        fig.update_layout(
        height=480,
        plot_bgcolor="rgba(75,45,95,0)",
        paper_bgcolor="rgba(75,45,95,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title_text="Cluster",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

        fig.update_traces(marker=dict(size=5, line=dict(width=0)))       

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- METRIC CARDS ----------
    with metric_col:
        st.markdown("### Model Quality Indicators")

        sil = silhouette_score(X_pca, clusters)
        db = davies_bouldin_score(X_pca, clusters)
        ch = calinski_harabasz_score(X_pca, clusters)

        def metric_card(title, value, progress, note, bg):
            st.markdown(f"""
            <div style="
                background:{bg};
                padding:18px;
                border-radius:16px;
                margin-bottom:16px;
                border:1px solid #E5E7EB;
                box-shadow:0 6px 14px rgba(0,0,0,0.06);
            ">
                <div style="font-weight:700;font-size:15px;margin-bottom:8px;">
                    {title}
                </div>
                <div style="font-size:26px;font-weight:800;color:#4C1D95;">
                    {value}
                </div>
                <div style="margin-top:6px;font-size:13px;color:#374151;">
                    {note}
                </div>
                <div style="
                    height:8px;
                    background:#E5E7EB;
                    border-radius:6px;
                    margin-top:10px;
                ">
                    <div style="
                        width:{progress}%;
                        height:100%;
                        background:#8B5CF6;
                        border-radius:6px;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        metric_card(
            "Silhouette Score",
            f"{sil:.3f}",
            min(sil * 100, 100),
            "🔴 < 0.2 = Weak | 🟡 0.2–0.5 = Moderate | 🟢 > 0.5 = Strong",
            "#EAD0FB"
        )

        metric_card(
            "Davies–Bouldin Index",
            f"{db:.3f}",
            min((1 / db) * 100, 100),
            "🔴 > 2.5 = Weak | 🟡 1.5–2.5 = Moderate | 🟢 < 1.5 = Strong",
            "#D4A2F6"
        )

        metric_card(
            "Calinski–Harabasz Index",
            f"{int(ch):,}",
            min(ch / 10000 * 100, 100),
            "🔴 < 3000 = Weak | 🟡 3000 – 7000 = Moderate | 🟢 > 7000 = Strong",
            "#BF73F2"
        )

    # -----------------------------------------------------
    # CLUSTER INSIGHTS CARDS
    # -----------------------------------------------------
    st.markdown("### 💡 Cluster Insights")

    if model_choice == "Gaussian Mixture Model (GMM)":
        display_profile = profile.sort_values(
            by="monetary", ascending=False
        ).head(4)
    else:
        display_profile = profile

    insight_cols = st.columns(len(display_profile))


    for i, row in display_profile.iterrows():
        freq = row["frequency"]
        rec = row["recency"]
        mon = row["monetary"]

        if mon > profile["monetary"].quantile(0.75):
            segment = "High Value"
            bg = "linear-gradient(135deg, #DCFCE7, #BBF7D0)"
            icon = "⭐"
        elif rec > profile["recency"].quantile(0.75):
            segment = "At Risk"
            bg = "linear-gradient(135deg, #FFE4E6, #FECDD3)"
            icon = "⚠️"
        else:
            segment = "Average"
            bg = "linear-gradient(135deg, #EDE9FE, #DDD6FE)"
            icon = "👤"

        insight = f"Avg spend ₹{mon:,.0f}, recency {rec:.0f} days, frequency {freq:.1f}"

        with insight_cols[i]:
            st.markdown(
                f"""
                <div style="
                    background:{bg};
                    border-radius:18px;
                    padding:18px;
                    box-shadow:0 6px 14px rgba(0,0,0,0.06);
                    height:100%;
                ">
                    <div style="font-size:22px;">{icon} Cluster {int(row['cluster'])}</div>
                    <div style="font-weight:600;margin-top:6px;">Segment: {segment}</div>
                    <div style="margin-top:10px;color:#374151;font-size:14px;">
                        {insight}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# CUSTOMER ANALYSIS PAGE
# =========================================================
elif page == "Customer Analysis":

    if "data" not in st.session_state:
        st.warning("Please upload a dataset on the Home page first.")
        st.stop()

    df = st.session_state["data"].copy()

    st.markdown("## 👥 Customer Analysis")
    st.caption("Understand customer value, loyalty, and engagement using RFM analysis.")

    # -----------------------------------------------------
    # RFM SCORING
    # -----------------------------------------------------
    r_labels = [4, 3, 2, 1]
    f_labels = [1, 2, 3, 4]
    m_labels = [1, 2, 3, 4]

    df["R"] = pd.qcut(df["recency"], 4, labels=r_labels)
    df["F"] = pd.qcut(df["frequency"].rank(method="first"), 4, labels=f_labels)
    df["M"] = pd.qcut(df["monetary"], 4, labels=m_labels)

    df["RFM_Score"] = df["R"].astype(str) + df["F"].astype(str) + df["M"].astype(str)

    def rfm_segment(row):
        if row["R"] >= 3 and row["F"] >= 3 and row["M"] >= 3:
            return "Champions"
        elif row["F"] >= 3:
            return "Loyal Customers"
        elif row["R"] <= 2 and row["F"] <= 2:
            return "At Risk"
        else:
            return "Potential"

    df["RFM_Segment"] = df.apply(rfm_segment, axis=1)
    st.session_state["data"] = df

    # -----------------------------------------------------
    # KPI STRIP
    # -----------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">Total Customers</div>
                <div class="kpi-value">{len(df):,}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">Champions %</div>
                <div class="kpi-value">
                    {(df['RFM_Segment'].eq('Champions').mean()*100):.1f}%
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">At Risk %</div>
                <div class="kpi-value">
                    {(df['RFM_Segment'].eq('At Risk').mean()*100):.1f}%
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">Avg RFM Score</div>
                <div class="kpi-value">
                    {df[['R','F','M']].astype(int).sum(axis=1).mean():.2f}
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    

    # =========================================================
    # RFM SEGMENT DISTRIBUTION + CUSTOMER VALUE MATRIX
    # =========================================================

    st.markdown("## RFM & Customer Value Overview ")

    left, right = st.columns(2, gap="large")

    # ---------------------------------------------------------
    # RFM DONUT CHART (LEFT CARD)
    # ---------------------------------------------------------
    with left:
        st.markdown("""
        <div class="card">
            <div class="card-title"> RFM Segment Distribution </div>
        """, unsafe_allow_html=True)

        SEGMENT_COLORS = {
            "Champions": "#22C55E",        # green
            "Loyal Customers": "#6366F1",  # indigo
            "Potential": "#FACC15",        # yellow
            "At Risk": "#EF4444"           # red
        }

        rfm_counts = df["RFM_Segment"].value_counts().reset_index()
        rfm_counts.columns = ["Segment", "Count"]

        fig_donut = px.pie(
            rfm_counts,
            values="Count",
            names="Segment",
            hole=0.55,
            color="Segment",
            color_discrete_map=SEGMENT_COLORS
        )

        fig_donut.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}"
        )

        fig_donut.update_layout(
            height=420,
            showlegend=True,
            margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"<b>{len(df):,}</b><br>Customers",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False
                )
            ]
        )

        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ---------------------------------------------------------
    # CUSTOMER VALUE MATRIX (RIGHT CARD)
    # ---------------------------------------------------------
    with right:
        st.markdown("""
        <div class="card">
            <div class="card-title"> Customer Value Matrix </div>
        """, unsafe_allow_html=True)

        df["Value_Level"] = np.where(
            df["monetary"] >= df["monetary"].median(),
            "High Value",
            "Low Value"
        )

        df["Engagement_Level"] = np.where(
            df["frequency"] >= df["frequency"].median(),
            "High Engagement",
            "Low Engagement"
        )

        matrix = (
            df
            .groupby(["Value_Level", "Engagement_Level"])
            .size()
            .reset_index(name="Customers")
        )

        pivot_matrix = matrix.pivot(
            index="Value_Level",
            columns="Engagement_Level",
            values="Customers"
        ).fillna(0)

        fig_heat = px.imshow(
            pivot_matrix,
            text_auto=True,
            color_continuous_scale=[
                "#EF4444",  # red
                "#FACC15",  # yellow
                "#22C55E"   # green
            ],
            labels=dict(color="Customers")
        )

        fig_heat.update_layout(
            height=420,
            font=dict(size=14),
            margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Engagement Level",
            yaxis_title="Customer Value"
        )

        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        
    st.markdown("### RFM Customer Table")

    DISPLAY_ROWS = 1000   


    def style_rfm_table(df):
        return (
            df
            .style
            .set_properties(**{
                "background-color": "#FFFFFF",
                "color": "#111827",
                "border-color": "#DDD6FE"
            })
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#6D28D9"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("font-size", "14px")
                ]},
                {"selector": "tbody tr:nth-child(even)", "props": [
                    ("background-color", "#F5F3FF")
                ]}
            ])
        )
    rfm_display = df[["recency", "frequency", "monetary","RFM_Score", "RFM_Segment"]].head(DISPLAY_ROWS)

    st.caption(f"Showing first {DISPLAY_ROWS:,} rows")
    st.dataframe(
            style_rfm_table(rfm_display),
            use_container_width=True,
            height=380
        )
    
    segment_profile = (
        df.groupby("RFM_Segment")[["recency", "frequency", "monetary"]]
        .mean()
        .reset_index()
    )
    

    # =========================================================
    # SEGMENT BEHAVIORAL SIGNAL CARDS (Gradient Style – FIXED)
    # =========================================================

    st.markdown("### Segment Behavioral Signals")
    st.caption("High-level behavioral patterns derived from RFM dimensions.")

    # Helper to convert score → %
    def pct(v):
        return f"{int(v * 100)}%"

    segment_metrics = (
        df.groupby("RFM_Segment")
        .agg(
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean")
        )
        .reset_index()
    )

    # Normalize for bars
    segment_metrics["recency_score"] = 1 - (segment_metrics["avg_recency"] / segment_metrics["avg_recency"].max())
    segment_metrics["frequency_score"] = segment_metrics["avg_frequency"] / segment_metrics["avg_frequency"].max()
    segment_metrics["monetary_score"] = segment_metrics["avg_monetary"] / segment_metrics["avg_monetary"].max()
    st.session_state["segment_metrics"] = segment_metrics.copy()


    SEGMENT_STYLE = {
        "Champions": {"bg": "#DCFCE7", "icon": "⭐"},
        "Loyal Customers": {"bg": "#E0E7FF", "icon": "💜"},
        "Potential": {"bg": "#FEF3C7", "icon": "🚀"},
        "At Risk": {"bg": "#FEE2E2", "icon": "⚠️"}
    }
    cols = st.columns(len(segment_metrics))

    for col, (_, row) in zip(cols, segment_metrics.iterrows()):
        style = SEGMENT_STYLE.get(row["RFM_Segment"], {"bg": "#F3F4F6", "icon": "👤"})

        with col:
            st.markdown(
    f"""
    <div style="background:{style['bg']};padding:14px;border-radius:16px;
    box-shadow:0 8px 18px rgba(0,0,0,0.08);height:100%;">

    <div style="font-size:16px;font-weight:800;margin-bottom:10px;">
        {style['icon']} {row['RFM_Segment']}
    </div>

    <div style="margin-bottom:8px;">
        <b style="font-size:13px;">Engagement</b>
        <div style="background:#E5E7EB;border-radius:6px;height:8px;">
        <div style="width:{int(row['frequency_score']*100)}%;
        background:#6366F1;height:8px;border-radius:6px;"></div>
        </div>
    </div>

    <div style="margin-bottom:8px;">
        <b style="font-size:13px;">Spend Intensity</b>
        <div style="background:#E5E7EB;border-radius:6px;height:8px;">
        <div style="width:{int(row['monetary_score']*100)}%;
        background:#22C55E;height:8px;border-radius:6px;"></div>
        </div>
    </div>

    <div>
        <b style="font-size:13px;">Freshness</b>
        <div style="background:#E5E7EB;border-radius:6px;height:8px;">
        <div style="width:{int(row['recency_score']*100)}%;
        background:#F59E0B;height:8px;border-radius:6px;"></div>
        </div>
    </div>

    </div>
    """,
                unsafe_allow_html=True
            )




# =========================================================
# ACTION BOARD PAGE
# =========================================================

elif page == "Action Board":

    st.markdown("## 🚀 Action Board")
    st.caption(
        "Operational recommendations derived from customer behavior, "
        "risk signals, and revenue impact."
    )

    df = st.session_state.get("data")
    if df is None:
        st.warning("Please upload data on the Home page.")
        st.stop()

    # =====================================================
    # 1️⃣ ACTION KPI STRIP
    # =====================================================
    at_risk_df = df[df["recency"] > df["recency"].quantile(0.75)]
    upsell_df = df[
        (df["frequency"] > df["frequency"].median()) &
        (df["monetary"] > df["monetary"].median())
    ]

    k1, k2, k3, k4 = st.columns(4)

    def kpi_card(title, value, bg, icon):
        st.markdown(
            f"""
            <div style="
                background:{bg};
                padding:26px;
                border-radius:24px;
                box-shadow:0 14px 34px rgba(0,0,0,0.12);
                text-align:center;
            ">
                <div style="
                    width:42px;height:42px;
                    border-radius:50%;
                    background:rgba(255,255,255,0.65);
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    margin:0 auto 10px auto;
                    font-size:18px;
                    font-weight:800;
                ">
                    {icon}
                </div>
                <div style="
                    font-size:13px;
                    font-weight:700;
                    letter-spacing:0.06em;
                    text-transform:uppercase;
                ">
                    {title}
                </div>
                <div style="font-size:32px;font-weight:900;margin-top:6px;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k1:
        kpi_card("Targetable Customers", f"{len(df):,}", "#EEF2FF", "👥")
    with k2:
        kpi_card("Segments in Focus", df["RFM_Segment"].nunique(), "#FDF4FF", "🎯")
    with k3:
        kpi_card("At-Risk Customers", f"{len(at_risk_df):,}", "#FFE4E6", "⚠️")
    with k4:
        kpi_card(
            "Upsell Potential",
            f"₹{upsell_df['monetary'].sum():,.0f}",
            "#ECFDF5",
            "₹"
        )

    st.markdown("---")
    
    priority_df = (
        df.groupby("RFM_Segment")
        .agg(
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean")
        )
        .reset_index()
    )

    priority_df["priority_index"] = (
        0.45 * (priority_df["avg_recency"] / priority_df["avg_recency"].max()) +
        0.35 * (priority_df["avg_monetary"] / priority_df["avg_monetary"].max()) +
        0.20 * (priority_df["avg_frequency"] / priority_df["avg_frequency"].max())
    )

    priority_df["priority_score"] = (priority_df["priority_index"] * 100).round(0)

      
        
    st.markdown("### Priority Signals")
    st.caption("Relative execution priority across customer segments.")

    colors = {
        "Champions": "#7C3AED",
        "Loyal Customers": "#F97316",
        "Potential": "#22C55E",
        "At Risk": "#EF4444"
    }

    cols = st.columns(4)

    for col, (_, row) in zip(cols, priority_df.iterrows()):
        with col:
            fig = go.Figure(data=[
                go.Pie(
                    values=[row["priority_score"], 100 - row["priority_score"]],
                    hole=0.7,
                    marker=dict(colors=[
                        colors.get(row["RFM_Segment"], "#A855F7"),
                        "#E5E7EB"
                    ]),
                    textinfo="none"
                )
            ])

            fig.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=200,
                annotations=[
                    dict(
                        text=f"<b>{int(row['priority_score'])}%</b>",
                        x=0.5,
                        y=0.55,
                        font_size=22,
                        showarrow=False
                    ),
                    dict(
                        text=row["RFM_Segment"],
                        x=0.5,
                        y=0.35,
                        font_size=12,
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)


    # =====================================================
    # 🎯 CAMPAIGN RECOMMENDATION (LEFT-ALIGNED CARD)
    # =====================================================

    st.markdown("### Campaign Recommendation")

    left_col, right_col = st.columns([2, 1.2], gap="large")

    # -------------------------------
    # LEFT: Campaign Card
    # -------------------------------
    with left_col:

        selected_segment = st.selectbox(
            "Select Customer Segment",
            ["Champions", "Loyal Customers", "Potential", "At Risk"]
        )

        CAMPAIGNS = {
            "Champions": {
                "title": "Loyalty Rewards Boost",
                "icon": "🏆",
                "actions": ["Exclusive rewards", "Early-access offers"],
                "color": "#DCFCE7"
            },
            "Loyal Customers": {
                "title": "Upsell Booster",
                "icon": "📈",
                "actions": ["Premium bundles", "Membership upgrades"],
                "color": "#E0F2FE"
            },
            "Potential": {
                "title": "Activation Push",
                "icon": "🚀",
                "actions": ["Personalized nudges", "Limited-time offers"],
                "color": "#FEF3C7"
            },
            "At Risk": {
                "title": "Churn Rescue",
                "icon": "🚨",
                "actions": ["Win-back discounts", "Reminder campaigns"],
                "color": "#FEE2E2"
            }
        }

        segment_metrics = st.session_state.get("segment_metrics")

        if segment_metrics is None:
            st.warning("Please visit the Customer Analysis page first.")
            st.stop()

        row = segment_metrics[
            segment_metrics["RFM_Segment"] == selected_segment
        ].iloc[0]

        # Safe normalization
        freshness = max(0, min(1, row["recency_score"]))
        value_score = max(0, min(1, row["monetary_score"]))

        retention_uplift = int(freshness * 25)
        revenue_uplift = int(value_score * 20)

        urgency = (
            "HIGH" if freshness < 0.35
            else "MEDIUM" if freshness < 0.65
            else "LOW"
        )


      
        cfg = CAMPAIGNS[selected_segment]
        components.html(
            f"""
            <div style="
                background:#D4A2F6;
                padding:26px;
                border-radius:22px;
                box-shadow:0 10px 24px rgba(0,0,0,0.08);
                font-family: Arial, sans-serif;
            ">

                <h3>🏆 {cfg['title']}</h3>
                <p><b>Target Segment:</b> {selected_segment}</p>

                <hr>

                <b>Recommended Actions</b>
                <ul>
                    <li>{cfg['actions'][0]}</li>
                    <li>{cfg['actions'][1]}</li>
                </ul>

                <hr>

                <div style="
                    background:#E9D5FF;
                    padding:14px 18px;
                    border-radius:14px;
                    width:fit-content;
                ">
                    <b>Expected Impact</b><br>

                    <span style="color:#15803D; font-weight:600;">
                        ↑ Retention +{retention_uplift}%
                    </span><br>

                    <span style="color:#15803D; font-weight:600;">
                        ↑ Revenue +{revenue_uplift}%
                    </span>
                </div>

                <hr>

                <b>Urgency:</b>
                <span style="
                    font-weight:700;
                    color:{'red' if urgency=='HIGH' else 'orange' if urgency=='MEDIUM' else 'green'};
                ">
                    {urgency}
                </span>

            </div>
            """,
            height=400,
        )


        

    # -------------------------------
    # RIGHT: ROI PLACEHOLDER
    # -------------------------------  


    CAMPAIGN_CONFIG = {
        "Churn Rescue": {
            "segment": "At Risk",
            "response_rate": 0.08,
            "uplift": 0.08,
            "cost_per_customer": 20
        },
        "Upsell Booster": {
            "segment": "Loyal Customers",
            "response_rate": 0.18,
            "uplift": 0.15,
            "cost_per_customer": 25
        },
        "Activation Push": {
            "segment": "Potential",
            "response_rate": 0.12,
            "uplift": 0.12,
            "cost_per_customer": 15
        }
    }
    with right_col:

        st.markdown("### Expected ROI")

        selected_campaign = st.selectbox(
            "Select Campaign",
            list(CAMPAIGN_CONFIG.keys())
        )

        cfg = CAMPAIGN_CONFIG[selected_campaign]

        seg_df = df[df["RFM_Segment"] == cfg["segment"]]

        if seg_df.empty:
            st.warning("No data available for this segment.")
            st.stop()

        # --- Calculations ---
        target_n = len(seg_df)
        avg_monetary = seg_df["monetary"].mean()

        responders = target_n * cfg["response_rate"]
        incremental_revenue = responders * avg_monetary * cfg["uplift"]
        cost = target_n * cfg["cost_per_customer"]

        roi_pct = ((incremental_revenue - cost) / cost) * 100

        # Confidence
        if target_n > 50000:
            confidence = "High"
            conf_color = "green"
        elif target_n > 20000:
            confidence = "Medium"
            conf_color = "orange"
        else:
            confidence = "Low"
            conf_color = "red"

        # ---------------------------
        # ROI CARD UI
        # ---------------------------
        roi_html = f"""
        <div style="background:#EAD0FB; padding:24px; border-radius:20px;
                    box-shadow:0 8px 20px rgba(0,0,0,0.08); ">

        <h3>📈 {selected_campaign}</h3>
        <p><b>Target Segment:</b> {cfg['segment']}</p>

        <hr>

        <b>Estimated Impact</b><br>
        <span style="color:green;">⬆ Revenue: ₹{incremental_revenue:,.0f}</span><br>
        <span style="color:blue;">ROI: {roi_pct:.1f}%</span>

        <hr>

        <b>Confidence</b><br>
        <span style="font-weight:700; color:{conf_color};">
        {confidence}
        </span>

        </div>
        """

        st.markdown(roi_html, unsafe_allow_html=True)
        