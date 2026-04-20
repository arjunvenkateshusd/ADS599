"""
app_streamlit.py
----------------
Single-Engine Predictive Maintenance Inference Tool
ADS-599 Capstone — Arjun Venkatesh, Duy-Anh Dang, Jorge Roldan

Run:
    python -m streamlit run app_streamlit.py
Or:
    python -m streamlit run ../code_library/app_streamlit.py 
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_library.data_preparation import (
    load_cmapss, load_naval,
    CMAPSS_FEATURE_COLS, NAVAL_FEATURE_COLS, NAVAL_TARGET_COLS,
)
from code_library.preprocessing import (
    split_engines, make_window_features, get_test_window_features,
    make_tabular, get_test_tabular, scale_naval,
)
from code_library.evaluation import evaluate_cmapss, evaluate_naval, rmse, mae, r2

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PredMaint · Inference Tool",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0d1b2a; color: #e8edf3; }
[data-testid="stSidebar"] { background: #111d2c; border-right: 1px solid #1e3048; }
[data-testid="metric-container"] {
    background: #162032; border: 1px solid #1e3a5a;
    border-radius: 10px; padding: 14px 18px;
}
[data-testid="metric-container"] label { color: #7fa8cc !important; font-size: 0.78rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8edf3 !important; font-size: 1.9rem !important; font-weight: 700;
}
[data-baseweb="tab-list"] { background: #111d2c; border-radius: 8px; }
[data-baseweb="tab"] { color: #7fa8cc !important; }
[aria-selected="true"] { color: #4da6ff !important; border-bottom: 2px solid #4da6ff !important; }
[data-testid="stExpander"] { background: #162032; border-radius: 8px; border: 1px solid #1e3a5a; }
.alert-critical { background:#3d0b0b; border-left:4px solid #e63946; padding:10px 14px; border-radius:0 8px 8px 0; color:#fbb; }
.alert-warning  { background:#3d2200; border-left:4px solid #f4a261; padding:10px 14px; border-radius:0 8px 8px 0; color:#fdb; }
.alert-healthy  { background:#0b3d20; border-left:4px solid #2dc653; padding:10px 14px; border-radius:0 8px 8px 0; color:#bfd; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Colour palette & layout helpers
# ---------------------------------------------------------------------------

CLR = {
    "bg":     "#0d1b2a",
    "panel":  "#162032",
    "grid":   "#1e3a5a",
    "text":   "#d0e4f7",
    "accent": "#4da6ff",
    "amber":  "#f4a261",
    "green":  "#2dc653",
    "red":    "#e63946",
    "purple": "#a78bfa",
    "teal":   "#22d3ee",
}

_BASE_LAYOUT = dict(
    paper_bgcolor=CLR["panel"],
    plot_bgcolor=CLR["panel"],
    font=dict(color=CLR["text"], family="Inter, Segoe UI, sans-serif"),
)


def pl(**overrides) -> dict:
    """Return a Plotly layout dict: base theme merged with per-call overrides."""
    layout = {**_BASE_LAYOUT}
    layout.update(overrides)
    # Ensure axes get grid colour if not explicitly set
    for ax in ("xaxis", "yaxis"):
        if ax not in layout:
            layout[ax] = {}
        if "gridcolor" not in layout[ax]:
            layout[ax]["gridcolor"] = CLR["grid"]
        if "linecolor" not in layout[ax]:
            layout[ax]["linecolor"] = CLR["grid"]
    return layout


WINDOW   = 30
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# Model training (cached once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training models on raw data …")
def build_cmapss_models(subset: str):
    raw_dir = os.path.join(DATA_DIR, "CMaps")
    train_df, test_df = load_cmapss(subset=subset, raw_dir=raw_dir, rul_clip=125)
    feat_cols = [c for c in train_df.columns if c not in ("unit_id", "cycle", "rul")]
    df_tr, df_val = split_engines(train_df, val_fraction=0.20, random_state=42)

    Xtr, ytr = make_window_features(df_tr,  feat_cols, window_size=WINDOW, step=5)
    Xva, yva = make_window_features(df_val, feat_cols, window_size=WINDOW, step=1)
    Xte, yte = get_test_window_features(test_df, feat_cols, window_size=WINDOW)

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xva_sc = scaler.transform(Xva)
    Xte_sc = scaler.transform(Xte)

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=25, verbosity=0,
    )
    model.fit(Xtr_sc, ytr, eval_set=[(Xva_sc, yva)], verbose=False)

    stat_names     = ["last", "mean", "std", "slope"]
    win_feat_names = [f"{f}_{s}" for f in feat_cols for s in stat_names]

    return dict(
        train_df=train_df, test_df=test_df, feat_cols=feat_cols,
        model=model, scaler=scaler,
        Xte_sc=Xte_sc, yte=yte,
        win_feat_names=win_feat_names,
    )


@st.cache_resource(show_spinner="Training Naval models …")
def build_naval_models():
    raw_dir = os.path.join(DATA_DIR, "UCI CBM Dataset")
    X_df, y_df = load_naval(raw_dir=raw_dir)
    X = X_df.values.astype(np.float32)
    y = y_df.values.astype(np.float32)

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]

    X_tr_sc, X_te_sc, scaler = scale_naval(X[tr], X[te])

    models = {}
    for i, tgt in enumerate(NAVAL_TARGET_COLS):
        m = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        m.fit(X_tr_sc, y[tr, i])
        models[tgt] = m

    return dict(
        X_df=X_df, y_df=y_df, scaler=scaler, models=models,
        X_tr_sc=X_tr_sc, X_te_sc=X_te_sc,
        y_tr=y[tr], y_te=y[te],
    )

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _window_row(feats: np.ndarray, feat_cols: list, scaler: StandardScaler):
    """Compute a single 72-feature window row from the last WINDOW cycles."""
    n = len(feats)
    w = feats[-WINDOW:] if n >= WINDOW else np.vstack(
        [np.tile(feats[0], (WINDOW - n, 1)), feats])
    t = np.arange(WINDOW, dtype=np.float32); t -= t.mean()
    denom = float((t ** 2).sum())
    mu = w.mean(0); si = w.std(0)
    sl = (t[:, None] * (w - mu)).sum(0) / denom
    row = np.empty(4 * len(feat_cols), dtype=np.float32)
    row[0::4] = w[-1]; row[1::4] = mu; row[2::4] = si; row[3::4] = sl
    return scaler.transform(row.reshape(1, -1))


def rul_status(rul_val: float, threshold: int) -> tuple:
    if rul_val < threshold * 0.5:
        return "CRITICAL — Schedule Immediately", "alert-critical"
    if rul_val < threshold:
        return "WARNING — Plan Maintenance", "alert-warning"
    return "HEALTHY — No Action Required", "alert-healthy"


def sensor_fig(engine_df: pd.DataFrame, feat_cols: list) -> go.Figure:
    cols  = feat_cols[:9]
    n     = len(cols)
    rows  = (n + 2) // 3
    fig   = make_subplots(rows=rows, cols=3, shared_xaxes=True,
                          subplot_titles=cols, vertical_spacing=0.07, horizontal_spacing=0.06)
    pal   = [CLR["accent"], CLR["amber"], CLR["green"], CLR["purple"],
             CLR["red"], CLR["teal"], "#f4a261", "#a78bfa", "#4da6ff"]
    for i, col in enumerate(cols):
        r, c = divmod(i, 3)
        fig.add_trace(go.Scatter(
            x=engine_df["cycle"], y=engine_df[col], mode="lines",
            line=dict(color=pal[i % len(pal)], width=1.8), showlegend=False,
        ), row=r + 1, col=c + 1)
    fig.update_layout(**pl(height=320 * rows, title="Sensor Degradation Profiles",
                           margin=dict(l=40, r=20, t=50, b=30)))
    for ax in fig.layout:
        if ax.startswith(("xaxis", "yaxis")):
            fig.layout[ax].update(gridcolor=CLR["grid"], linecolor=CLR["grid"])
    return fig


def rul_timeline_fig(engine_df: pd.DataFrame, model, scaler, feat_cols: list) -> go.Figure:
    feats  = engine_df[feat_cols].values.astype(np.float32)
    labels = engine_df["rul"].values.astype(np.float32)
    cycles = engine_df["cycle"].values
    n      = len(feats)
    preds  = np.full(n, np.nan)
    t      = np.arange(WINDOW, dtype=np.float32); t -= t.mean()
    denom  = float((t ** 2).sum())
    for end in range(WINDOW, n + 1):
        w   = feats[end - WINDOW: end]
        mu  = w.mean(0); si = w.std(0)
        sl  = (t[:, None] * (w - mu)).sum(0) / denom
        row = np.empty(4 * len(feat_cols), dtype=np.float32)
        row[0::4] = w[-1]; row[1::4] = mu; row[2::4] = si; row[3::4] = sl
        preds[end - 1] = float(model.predict(scaler.transform(row.reshape(1, -1)))[0])
    preds = np.clip(preds, 0, None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cycles, y=labels, name="True RUL",
                             line=dict(color=CLR["green"], width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=cycles, y=preds, name="Predicted RUL",
                             line=dict(color=CLR["accent"], width=2.5, dash="dash"), mode="lines"))
    fig.add_hrect(y0=0, y1=20, fillcolor=CLR["red"],   opacity=0.12,
                  annotation_text="Critical", annotation_position="top right")
    fig.add_hrect(y0=20, y1=50, fillcolor=CLR["amber"], opacity=0.08,
                  annotation_text="Warning",  annotation_position="top right")
    fig.update_layout(**pl(
        title="RUL Timeline — True vs Predicted",
        xaxis_title="Engine Cycle", yaxis_title="RUL (cycles)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=50, b=40),
    ))
    return fig


def shap_waterfall_fig(model, x_row_sc: np.ndarray, win_feat_names: list, n_top: int = 15) -> go.Figure:
    explainer = shap.TreeExplainer(model)
    sv   = explainer.shap_values(x_row_sc.reshape(1, -1))[0]
    base = float(explainer.expected_value)
    df_s = pd.DataFrame({"feature": win_feat_names, "shap": sv})
    df_s = df_s.reindex(df_s["shap"].abs().sort_values(ascending=False).index).head(n_top)
    df_s = df_s.sort_values("shap")
    colors = [CLR["green"] if v > 0 else CLR["red"] for v in df_s["shap"]]
    fig = go.Figure(go.Bar(
        x=df_s["shap"], y=df_s["feature"], orientation="h",
        marker_color=colors, text=[f"{v:+.2f}" for v in df_s["shap"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color=CLR["text"], line_width=1)
    fig.update_layout(**pl(
        title=f"SHAP Feature Impact  (base = {base:.1f} cycles)",
        xaxis_title="SHAP value (impact on RUL prediction)",
        height=420, margin=dict(l=160, r=50, t=50, b=40),
    ))
    return fig


def parity_fig(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> go.Figure:
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name=name,
                             marker=dict(color=CLR["accent"], opacity=0.5, size=5)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=CLR["red"], dash="dash"), name="Perfect"))
    fig.update_layout(**pl(
        title=f"{name} — Predicted vs Actual",
        xaxis_title="Actual RUL", yaxis_title="Predicted RUL",
        margin=dict(l=50, r=20, t=50, b=40),
    ))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔧 PredMaint")
    st.markdown('<p style="color:#7fa8cc;font-size:0.75rem;">ADS-599 Capstone · USD · 2026</p>',
                unsafe_allow_html=True)
    st.divider()

    domain = st.radio("**Domain**", ["CMAPSS (Turbofan Engine)", "Naval Propulsion"])

    st.divider()
    if domain.startswith("CMAPSS"):
        subset = st.selectbox("**CMAPSS Subset**",
                              ["FD001", "FD002", "FD003", "FD004"],
                              help="FD001/FD003 = 1 op condition; FD002/FD004 = 6 conditions")
        rul_threshold = st.slider("**Alert Threshold (RUL cycles)**",
                                  min_value=10, max_value=100, value=30, step=5)
    else:
        target        = st.radio("**Target**", ["kMc (Compressor)", "kMt (Turbine)"])
        rul_threshold = 30

    st.divider()
    st.markdown('<p style="color:#4a6a8a;font-size:0.72rem;">Models train from raw data on first load. Refresh page to retrain.</p>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown("# Predictive Maintenance · Single-Engine Inference")
st.markdown("*Gradient Boosting + Window Feature Engineering for RUL Estimation*")
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# CMAPSS DOMAIN
# ══════════════════════════════════════════════════════════════════════════
if domain.startswith("CMAPSS"):
    with st.spinner(f"Loading {subset} …"):
        c = build_cmapss_models(subset)

    train_df  = c["train_df"]; test_df = c["test_df"]
    feat_cols = c["feat_cols"]; model   = c["model"]
    scaler    = c["scaler"];    Xte_sc  = c["Xte_sc"]
    yte       = c["yte"];       wfn     = c["win_feat_names"]

    tab1, tab2, tab3 = st.tabs(["🔍  Single Engine", "📊  Fleet Snapshot", "🧠  Model Intelligence"])

    # ── Tab 1: Single Engine ───────────────────────────────────────────────
    with tab1:
        engines   = sorted(train_df["unit_id"].unique())
        engine_id = st.selectbox("Select engine", engines)
        eng_df    = train_df[train_df["unit_id"] == engine_id].sort_values("cycle")

        feats    = eng_df[feat_cols].values.astype(np.float32)
        row_sc   = _window_row(feats, feat_cols, scaler)
        pred_rul = float(np.clip(model.predict(row_sc)[0], 0, None))
        true_rul = float(eng_df["rul"].iloc[-1])
        error    = pred_rul - true_rul

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted RUL",       f"{pred_rul:.0f} cycles")
        c2.metric("True RUL",            f"{true_rul:.0f} cycles")
        c3.metric("Error",               f"{error:+.1f} cycles",
                  delta="Early" if error > 0 else "Late", delta_color="inverse")
        c4.metric("Cycles Observed",     f"{len(eng_df)}")

        label, css = rul_status(pred_rul, rul_threshold)
        st.markdown(f'<div class="{css}"><b>{label}</b> — Predicted RUL: {pred_rul:.0f} cycles (threshold: {rul_threshold})</div>',
                    unsafe_allow_html=True)
        st.markdown("")

        st.plotly_chart(sensor_fig(eng_df, feat_cols), use_container_width=True)

        with st.spinner("Computing rolling RUL predictions …"):
            st.plotly_chart(rul_timeline_fig(eng_df, model, scaler, feat_cols),
                            use_container_width=True)

        st.subheader("Model Explanation (SHAP)")
        with st.spinner("Computing SHAP values …"):
            st.plotly_chart(shap_waterfall_fig(model, row_sc[0], wfn),
                            use_container_width=True)

        with st.expander("What are SHAP values?"):
            st.markdown(f"""
SHAP (SHapley Additive exPlanations) decomposes each prediction into per-feature contributions:
- **Green** → pushes RUL *higher* (sensor reading healthier than average)
- **Red** → pushes RUL *lower* (degradation signal detected)
- Base value = model's average RUL prediction across all test engines

Feature names: `{{sensor}}_{{stat}}` where stats are `last`, `mean`, `std`, `slope`
computed over the most recent **{WINDOW} cycles**.
""")

    # ── Tab 2: Fleet Snapshot ──────────────────────────────────────────────
    with tab2:
        preds_all  = np.clip(model.predict(Xte_sc), 0, None)
        n_critical = int((preds_all < rul_threshold * 0.5).sum())
        n_warning  = int(((preds_all >= rul_threshold * 0.5) & (preds_all < rul_threshold)).sum())
        n_healthy  = int((preds_all >= rul_threshold).sum())

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Fleet Size",          len(preds_all))
        cb.metric("Critical ⚠",          n_critical,  delta="Immediate",   delta_color="inverse")
        cc.metric("Warning",             n_warning,   delta="Plan maint.", delta_color="off")
        cd.metric("Healthy ✓",           n_healthy)

        # RUL histogram
        fig_h = go.Figure(go.Histogram(x=preds_all, nbinsx=28,
                                        marker_color=CLR["accent"], opacity=0.85))
        fig_h.add_vline(x=rul_threshold * 0.5, line_color=CLR["red"],
                        line_dash="dash", annotation_text="Critical", annotation_font_color=CLR["red"])
        fig_h.add_vline(x=rul_threshold, line_color=CLR["amber"],
                        line_dash="dash", annotation_text="Warning",  annotation_font_color=CLR["amber"])
        fig_h.update_layout(**pl(title="Fleet RUL Distribution",
                                 xaxis_title="Predicted RUL (cycles)", yaxis_title="Count",
                                 height=300, margin=dict(l=50, r=20, t=50, b=40)))
        st.plotly_chart(fig_h, use_container_width=True)

        # Fleet bar chart
        test_ids   = sorted(test_df["unit_id"].unique())[:len(preds_all)]
        bar_colors = [CLR["red"] if p < rul_threshold * 0.5
                      else CLR["amber"] if p < rul_threshold
                      else CLR["green"]
                      for p in preds_all]
        fig_fleet = go.Figure(go.Bar(
            x=[f"E{e}" for e in test_ids], y=preds_all,
            marker_color=bar_colors,
            text=[f"{p:.0f}" for p in preds_all], textposition="outside", textfont=dict(size=8),
        ))
        fig_fleet.add_hline(y=rul_threshold,       line_color=CLR["amber"], line_dash="dash")
        fig_fleet.add_hline(y=rul_threshold * 0.5, line_color=CLR["red"],   line_dash="dash")
        fig_fleet.update_layout(**pl(title="Engine-by-Engine Predicted RUL",
                                     xaxis_title="Engine", yaxis_title="Predicted RUL",
                                     height=300, margin=dict(l=50, r=20, t=50, b=40),
                                     xaxis=dict(tickfont=dict(size=8), gridcolor=CLR["grid"])))
        st.plotly_chart(fig_fleet, use_container_width=True)

        # Parity
        st.plotly_chart(parity_fig(yte, preds_all, f"Window-XGBoost ({subset})"),
                        use_container_width=True)

        # Priority table
        fleet_df = pd.DataFrame({
            "Engine ID":     test_ids,
            "Predicted RUL": preds_all.round(1),
            "True RUL":      yte.round(1),
            "Error":         (preds_all - yte).round(1),
            "Status":        ["Critical" if p < rul_threshold * 0.5
                               else "Warning" if p < rul_threshold
                               else "Healthy"
                               for p in preds_all],
        }).sort_values("Predicted RUL")

        def _color_status(val):
            return {"Critical": "background-color:#3d0b0b;color:#fbb",
                    "Warning":  "background-color:#3d2200;color:#fdb",
                    "Healthy":  "background-color:#0b3d20;color:#bfd"}.get(val, "")

        st.dataframe(
            fleet_df.style.applymap(_color_status, subset=["Status"])
                          .format({"Predicted RUL": "{:.1f}", "True RUL": "{:.1f}", "Error": "{:+.1f}"}),
            use_container_width=True, height=300,
        )

    # ── Tab 3: Model Intelligence ──────────────────────────────────────────
    with tab3:
        st.subheader("Global SHAP Feature Importance")
        with st.spinner("Computing SHAP on test set …"):
            try:
                explainer = shap.TreeExplainer(model)
                sv        = np.abs(np.array(explainer.shap_values(Xte_sc))).mean(axis=0)
                shap_df   = (pd.DataFrame({"feature": wfn, "mean_|SHAP|": sv})
                               .sort_values("mean_|SHAP|", ascending=False).head(20))
                fig_gs = go.Figure(go.Bar(
                    x=shap_df["mean_|SHAP|"], y=shap_df["feature"],
                    orientation="h", marker_color=CLR["purple"],
                ))
                fig_gs.update_layout(**pl(
                    title="Mean |SHAP| — Test Set",
                    xaxis_title="Mean |SHAP value|",
                    height=500, margin=dict(l=160, r=20, t=50, b=40),
                    yaxis=dict(autorange="reversed", gridcolor=CLR["grid"]),
                ))
                st.plotly_chart(fig_gs, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP skipped: {e}")

        # XGBoost gain importance
        st.subheader("XGBoost Feature Gain")
        imp = xgb_model_importance = model.get_booster().get_fscore()
        if imp:
            imp_df = (pd.DataFrame({"feature": list(imp.keys()), "gain": list(imp.values())})
                        .sort_values("gain", ascending=False).head(20))
            fig_gi = go.Figure(go.Bar(
                x=imp_df["gain"], y=imp_df["feature"],
                orientation="h", marker_color=CLR["accent"],
            ))
            fig_gi.update_layout(**pl(
                title="Top-20 Features by XGBoost Gain",
                xaxis_title="Gain",
                height=500, margin=dict(l=160, r=20, t=50, b=40),
                yaxis=dict(autorange="reversed", gridcolor=CLR["grid"]),
            ))
            st.plotly_chart(fig_gi, use_container_width=True)

        with st.expander("Window Feature Engineering"):
            st.markdown(f"""
For each **{WINDOW}-cycle** sliding window, **4 temporal statistics** are extracted per sensor:

| Stat | Meaning |
|------|---------|
| `last`  | Current sensor reading |
| `mean`  | 30-cycle average |
| `std`   | 30-cycle standard deviation |
| `slope` | Least-squares linear trend |

**{len(feat_cols)} sensors × 4 stats = {4*len(feat_cols)} engineered features** fed to XGBoost.
Temporal statistics encode the degradation *trajectory*, not just the current state.
""")

        with st.expander("Model Performance"):
            m = evaluate_cmapss(yte, np.clip(model.predict(Xte_sc), 0, None),
                                name=f"Window-XGBoost ({subset})", split="Test")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("RMSE",       f"{m['RMSE']:.2f}")
            mc2.metric("MAE",        f"{m['MAE']:.2f}")
            mc3.metric("R²",         f"{m['R²']:.4f}")
            mc4.metric("NASA Score", f"{m['NASA Score']:.0f}")


# ══════════════════════════════════════════════════════════════════════════
# NAVAL DOMAIN
# ══════════════════════════════════════════════════════════════════════════
else:
    with st.spinner("Loading Naval models …"):
        nav = build_naval_models()

    X_df    = nav["X_df"]; y_df   = nav["y_df"]
    scaler  = nav["scaler"]
    models  = nav["models"]
    X_te_sc = nav["X_te_sc"]; y_te = nav["y_te"]

    tgt_key = "kMc" if target.startswith("kMc") else "kMt"
    tgt_idx = 0 if tgt_key == "kMc" else 1

    tab1, tab2 = st.tabs(["📈  Degradation Analysis", "🧠  Model Intelligence"])

    with tab1:
        st.subheader(f"Naval Propulsion — {tgt_key} Degradation")

        preds   = models[tgt_key].predict(X_te_sc)
        y_1d    = y_te[:, tgt_idx]
        r_rmse  = rmse(y_1d, preds); r_mae = mae(y_1d, preds); r_r2 = r2(y_1d, preds)

        n1, n2, n3 = st.columns(3)
        n1.metric("RMSE", f"{r_rmse:.5f}")
        n2.metric("MAE",  f"{r_mae:.5f}")
        n3.metric("R²",   f"{r_r2:.5f}")

        st.plotly_chart(parity_fig(y_1d, preds, f"LightGBM — {tgt_key}"), use_container_width=True)

        # Correlation bar
        corr = pd.concat([X_df, y_df], axis=1).corr()
        tgt_corr = (corr[tgt_key].drop(NAVAL_TARGET_COLS, errors="ignore")
                                  .sort_values(key=abs, ascending=False))
        fig_c = go.Figure(go.Bar(
            x=tgt_corr.values, y=tgt_corr.index, orientation="h",
            marker_color=[CLR["green"] if v > 0 else CLR["red"] for v in tgt_corr.values],
        ))
        fig_c.update_layout(**pl(
            title=f"Pearson Correlation — Features vs {tgt_key}",
            height=420, margin=dict(l=210, r=20, t=50, b=40),
            yaxis=dict(gridcolor=CLR["grid"]),
        ))
        st.plotly_chart(fig_c, use_container_width=True)

    with tab2:
        st.subheader("LightGBM Feature Importance")
        fi    = models[tgt_key].feature_importances_
        fi_df = (pd.DataFrame({"feature": NAVAL_FEATURE_COLS, "importance": fi})
                   .sort_values("importance", ascending=False))
        fig_f = go.Figure(go.Bar(
            x=fi_df["importance"], y=fi_df["feature"],
            orientation="h", marker_color=CLR["accent"],
        ))
        fig_f.update_layout(**pl(
            title=f"LightGBM Gain — {tgt_key}",
            xaxis_title="Importance",
            height=440, margin=dict(l=210, r=20, t=50, b=40),
            yaxis=dict(autorange="reversed", gridcolor=CLR["grid"]),
        ))
        st.plotly_chart(fig_f, use_container_width=True)

        with st.expander("Dataset Overview"):
            st.markdown(f"""
**UCI Naval Propulsion Plants Dataset**
- **{len(X_df):,} steady-state operating points** from a physics-based gas turbine simulation
- **{len(NAVAL_FEATURE_COLS)} operational features**: lever position, shaft torque, temperatures, pressures …
- **Targets**: `kMc` (compressor decay) · `kMt` (turbine decay) — both ∈ [0.95, 1.0], higher = healthier
""")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    '<p style="color:#4a6a8a;font-size:0.72rem;text-align:center;">'
    "ADS-599 Capstone Thesis · University of San Diego · 2026 · "
    "Arjun Venkatesh · Duy-Anh Dang · Jorge Roldan"
    "</p>",
    unsafe_allow_html=True,
)
