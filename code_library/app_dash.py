"""
app_dash.py
-----------
Fleet Analytics Dashboard — Predictive Maintenance
ADS-599 Capstone — Arjun Venkatesh, Duy-Anh Dang, Jorge Roldan

Run:
    python app_dash.py
    # then open http://127.0.0.1:8050
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_library.data_preparation import (
    load_cmapss, load_naval,
    CMAPSS_FEATURE_COLS, NAVAL_FEATURE_COLS, NAVAL_TARGET_COLS,
)
from code_library.preprocessing import (
    split_engines, make_window_features, get_test_window_features, scale_naval,
)
from code_library.evaluation import evaluate_cmapss, evaluate_naval, rmse, mae, r2

# ---------------------------------------------------------------------------
# Constants & palette
# ---------------------------------------------------------------------------

WINDOW   = 30
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUBSETS  = ["FD001", "FD002", "FD003", "FD004"]

CLR = dict(
    bg     = "#0a1628",
    panel  = "#101f35",
    border = "#1a3050",
    text   = "#c8ddf0",
    muted  = "#5a7a9a",
    accent = "#4da6ff",
    amber  = "#f4a261",
    green  = "#2dc653",
    red    = "#e63946",
    purple = "#a78bfa",
    teal   = "#22d3ee",
)

_PLOT_BASE = dict(
    paper_bgcolor=CLR["panel"],
    plot_bgcolor=CLR["panel"],
    font=dict(color=CLR["text"], family="Inter, Segoe UI, sans-serif", size=12),
)


def pb(**overrides) -> dict:
    """Return Plotly layout dict: base theme merged with per-call overrides."""
    layout = {**_PLOT_BASE}
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Data & model loading (once at startup)
# ---------------------------------------------------------------------------

print("Loading data and training models …")
_cache = {}

for _subset in SUBSETS:
    _raw = os.path.join(DATA_DIR, "CMaps")
    _train_df, _test_df = load_cmapss(subset=_subset, raw_dir=_raw, rul_clip=125)
    _fc = [c for c in _train_df.columns if c not in ("unit_id", "cycle", "rul")]
    _df_tr, _df_val = split_engines(_train_df, val_fraction=0.20, random_state=42)

    _Xtr, _ytr = make_window_features(_df_tr,  _fc, window_size=WINDOW, step=5)
    _Xva, _yva = make_window_features(_df_val, _fc, window_size=WINDOW, step=1)
    _Xte, _yte = get_test_window_features(_test_df, _fc, window_size=WINDOW)

    _sc = StandardScaler()
    _Xtr_sc = _sc.fit_transform(_Xtr)
    _Xva_sc = _sc.transform(_Xva)
    _Xte_sc = _sc.transform(_Xte)

    _m = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=25, verbosity=0,
    )
    _m.fit(_Xtr_sc, _ytr, eval_set=[(_Xva_sc, _yva)], verbose=False)

    _preds = np.clip(_m.predict(_Xte_sc), 0, None)
    _metrics = evaluate_cmapss(_yte, _preds, name=f"Window-XGBoost ({_subset})", split="Test")

    _stat_names = ["last", "mean", "std", "slope"]
    _wfn = [f"{f}_{s}" for f in _fc for s in _stat_names]

    _cache[_subset] = dict(
        train_df=_train_df, test_df=_test_df, feat_cols=_fc,
        model=_m, scaler=_sc,
        Xte_sc=_Xte_sc, yte=_yte, preds_test=_preds,
        metrics=_metrics, win_feat_names=_wfn,
    )
    print(f"  {_subset}: RMSE={_metrics['RMSE']:.2f}  R²={_metrics['R²']:.4f}")

# Naval
_nav_raw = os.path.join(DATA_DIR, "UCI CBM Dataset")
_nav_X_df, _nav_y_df = load_naval(raw_dir=_nav_raw)
_nav_X = _nav_X_df.values.astype(np.float32)
_nav_y = _nav_y_df.values.astype(np.float32)

_rng = np.random.default_rng(42)
_idx = _rng.permutation(len(_nav_X))
_split = int(0.8 * len(_nav_X))
_tr_i, _te_i = _idx[:_split], _idx[_split:]

_nav_X_tr_sc, _nav_X_te_sc, _nav_scaler = scale_naval(_nav_X[_tr_i], _nav_X[_te_i])

_nav_models = {}
for _i, _tgt in enumerate(NAVAL_TARGET_COLS):
    _nm = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    _nm.fit(_nav_X_tr_sc, _nav_y[_tr_i, _i])
    _nav_models[_tgt] = _nm

_nav_preds = np.column_stack([_nav_models[t].predict(_nav_X_te_sc) for t in NAVAL_TARGET_COLS])
_nav_metrics = evaluate_naval(_nav_y[_te_i], _nav_preds, name="LightGBM", split="Test")
print(f"  Naval kMc R²={_nav_metrics['R²_kMc']:.5f}  kMt R²={_nav_metrics['R²_kMt']:.5f}")
print("Ready.\n")

# ---------------------------------------------------------------------------
# Rolling RUL helper
# ---------------------------------------------------------------------------

def rolling_rul(engine_df: pd.DataFrame, feat_cols: list, model, scaler) -> np.ndarray:
    feats = engine_df[feat_cols].values.astype(np.float32)
    n     = len(feats)
    preds = np.full(n, np.nan)
    t     = np.arange(WINDOW, dtype=np.float32); t -= t.mean()
    denom = float((t ** 2).sum())
    for end in range(WINDOW, n + 1):
        w  = feats[end - WINDOW: end]
        mu = w.mean(0); si = w.std(0)
        sl = (t[:, None] * (w - mu)).sum(0) / denom
        row = np.empty(4 * len(feat_cols), dtype=np.float32)
        row[0::4] = w[-1]; row[1::4] = mu; row[2::4] = si; row[3::4] = sl
        preds[end - 1] = float(model.predict(scaler.transform(row.reshape(1, -1)))[0])
    return np.clip(preds, 0, None)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Fleet PredMaint Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server

DD_STYLE = {
    "backgroundColor": CLR["panel"],
    "color":           CLR["text"],
    "border":          f"1px solid {CLR['border']}",
    "borderRadius":    "6px",
}


def kpi_card(value, label, color=None):
    color = color or CLR["accent"]
    return dbc.Card(
        dbc.CardBody([
            html.Div(str(value), style={
                "fontSize": "2rem", "fontWeight": "700",
                "color": color, "lineHeight": "1.1",
            }),
            html.Div(label, style={"color": CLR["muted"], "fontSize": "0.78rem", "marginTop": "6px"}),
        ]),
        style={
            "background": CLR["panel"], "border": f"1px solid {CLR['border']}",
            "borderRadius": "10px", "textAlign": "center",
        },
    )


navbar = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.Span("⚙️", style={"fontSize": "1.4rem", "marginRight": "10px"}),
            html.Span("Fleet PredMaint", style={
                "fontWeight": "700", "fontSize": "1.2rem",
                "color": CLR["accent"], "letterSpacing": "0.5px",
            }),
            html.Span(" · Fleet Analytics Dashboard", style={
                "color": CLR["muted"], "fontSize": "0.9rem", "marginLeft": "6px",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div("ADS-599 Capstone · USD 2026",
                 style={"color": CLR["muted"], "fontSize": "0.78rem"}),
    ], fluid=True, style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
    color=CLR["panel"],
    style={"borderBottom": f"1px solid {CLR['border']}", "padding": "10px 0"},
)

TAB_STYLE = {
    "backgroundColor": CLR["panel"], "color": CLR["muted"],
    "border": f"1px solid {CLR['border']}", "borderRadius": "6px 6px 0 0",
    "padding": "10px 20px", "fontWeight": "500",
}
TAB_ACTIVE = {**TAB_STYLE, "color": CLR["accent"],
              "borderBottom": f"2px solid {CLR['accent']}", "backgroundColor": CLR["bg"]}

app.layout = html.Div(
    style={"backgroundColor": CLR["bg"], "minHeight": "100vh",
           "color": CLR["text"], "fontFamily": "Inter, Segoe UI, sans-serif"},
    children=[
        navbar,
        dbc.Container(fluid=True, style={"padding": "20px"}, children=[
            # Controls row
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label("Subset", style={"color": CLR["muted"], "fontSize": "0.8rem"}),
                    dcc.Dropdown(id="subset-select",
                                 options=[{"label": s, "value": s} for s in SUBSETS],
                                 value="FD001", clearable=False, style=DD_STYLE),
                ]), width=2),
                dbc.Col(html.Div([
                    html.Label("Alert Threshold (RUL ≤)", style={"color": CLR["muted"], "fontSize": "0.8rem"}),
                    dcc.Slider(id="alert-threshold", min=10, max=80, step=5, value=30,
                               marks={10: "10", 30: "30", 50: "50", 80: "80"},
                               tooltip={"placement": "bottom"}),
                ]), width=4),
                dbc.Col(html.Div([
                    html.Label("Engine (Deep Dive)", style={"color": CLR["muted"], "fontSize": "0.8rem"}),
                    dcc.Dropdown(id="engine-select", clearable=False, style=DD_STYLE),
                ]), width=3),
            ], className="mb-3", align="center"),

            # Tabs
            dcc.Tabs(id="main-tabs", value="tab-fleet",
                     children=[
                         dcc.Tab(label="🗺  Fleet Overview",     value="tab-fleet",
                                 style=TAB_STYLE, selected_style=TAB_ACTIVE),
                         dcc.Tab(label="🔍  Engine Deep Dive",   value="tab-engine",
                                 style=TAB_STYLE, selected_style=TAB_ACTIVE),
                         dcc.Tab(label="🌊  Naval Domain",        value="tab-naval",
                                 style=TAB_STYLE, selected_style=TAB_ACTIVE),
                         dcc.Tab(label="📊  Cross-Domain Report", value="tab-report",
                                 style=TAB_STYLE, selected_style=TAB_ACTIVE),
                     ]),
            html.Div(id="tab-content", style={
                "backgroundColor": CLR["panel"],
                "border": f"1px solid {CLR['border']}",
                "borderTop": "none",
                "borderRadius": "0 0 10px 10px",
                "padding": "20px",
            }),
        ]),
        html.Footer(
            "ADS-599 Capstone · University of San Diego · 2026 · "
            "Arjun Venkatesh · Duy-Anh Dang · Jorge Roldan",
            style={"textAlign": "center", "color": CLR["muted"],
                   "fontSize": "0.72rem", "padding": "16px",
                   "borderTop": f"1px solid {CLR['border']}"},
        ),
    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("engine-select", "options"),
    Output("engine-select", "value"),
    Input("subset-select", "value"),
)
def update_engine_options(subset):
    engines = sorted(_cache[subset]["train_df"]["unit_id"].unique())
    return [{"label": f"Engine {e}", "value": e} for e in engines], engines[0]


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs",       "value"),
    Input("subset-select",   "value"),
    Input("alert-threshold", "value"),
    Input("engine-select",   "value"),
)
def render_tab(tab, subset, threshold, engine_id):
    c = _cache[subset]
    if tab == "tab-fleet":
        return fleet_layout(c, subset, threshold)
    if tab == "tab-engine":
        return engine_layout(c, subset, engine_id)
    if tab == "tab-naval":
        return naval_layout()
    if tab == "tab-report":
        return report_layout()
    return html.Div()

# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------

def fleet_layout(c, subset, threshold):
    preds   = c["preds_test"]
    yte     = c["yte"]
    metrics = c["metrics"]

    n_crit = int((preds < threshold * 0.5).sum())
    n_warn = int(((preds >= threshold * 0.5) & (preds < threshold)).sum())
    n_ok   = int((preds >= threshold).sum())

    kpi_row = dbc.Row([
        dbc.Col(kpi_card(len(preds),             "Engines"),                     width=2),
        dbc.Col(kpi_card(n_crit,                 "Critical  ⚠", CLR["red"]),     width=2),
        dbc.Col(kpi_card(n_warn,                 "Warning",      CLR["amber"]),   width=2),
        dbc.Col(kpi_card(n_ok,                   "Healthy  ✓",   CLR["green"]),   width=2),
        dbc.Col(kpi_card(f"{preds.mean():.0f}",  "Avg RUL"),                     width=2),
        dbc.Col(kpi_card(f"{metrics['RMSE']:.2f}","RMSE", CLR["purple"]),        width=2),
    ], className="mb-3")

    # Histogram
    fig_h = go.Figure(go.Histogram(x=preds, nbinsx=25,
                                   marker_color=CLR["accent"], opacity=0.85))
    fig_h.add_vline(x=threshold * 0.5, line_color=CLR["red"],
                    line_dash="dash", annotation_text="Critical",
                    annotation_font_color=CLR["red"])
    fig_h.add_vline(x=threshold, line_color=CLR["amber"],
                    line_dash="dash", annotation_text="Warning",
                    annotation_font_color=CLR["amber"])
    fig_h.update_layout(**pb(
        title="Fleet RUL Distribution",
        xaxis_title="Predicted RUL (cycles)", yaxis_title="Count",
        height=300, margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor=CLR["border"]), yaxis=dict(gridcolor=CLR["border"]),
    ))

    # Parity
    lo, hi = float(min(yte.min(), preds.min())), float(max(yte.max(), preds.max()))
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=yte, y=preds, mode="markers",
                               marker=dict(color=CLR["accent"], opacity=0.5, size=6), name="Engines"))
    fig_p.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                               line=dict(color=CLR["red"], dash="dash"), name="Perfect"))
    fig_p.update_layout(**pb(
        title="Predicted vs Actual RUL",
        xaxis_title="True RUL", yaxis_title="Predicted RUL",
        height=300, margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor=CLR["border"]), yaxis=dict(gridcolor=CLR["border"]),
    ))

    # Per-engine bar
    test_ids  = sorted(c["test_df"]["unit_id"].unique())[:len(preds)]
    bar_clrs  = [CLR["red"] if p < threshold * 0.5
                 else CLR["amber"] if p < threshold
                 else CLR["green"]
                 for p in preds]
    fig_bars = go.Figure(go.Bar(
        x=[f"E{e}" for e in test_ids], y=preds,
        marker_color=bar_clrs,
        text=[f"{p:.0f}" for p in preds], textposition="outside", textfont=dict(size=8),
    ))
    fig_bars.add_hline(y=threshold,       line_color=CLR["amber"], line_dash="dash")
    fig_bars.add_hline(y=threshold * 0.5, line_color=CLR["red"],   line_dash="dash")
    fig_bars.update_layout(**pb(
        title="Engine-by-Engine Predicted RUL",
        xaxis_title="Engine", yaxis_title="Predicted RUL (cycles)",
        height=300, margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor=CLR["border"], tickfont=dict(size=8)),
        yaxis=dict(gridcolor=CLR["border"]),
    ))

    # Priority table
    pri_df = pd.DataFrame({
        "Engine":   test_ids,
        "Pred RUL": np.round(preds, 1),
        "True RUL": np.round(yte,   1),
        "Error":    np.round(preds - yte, 1),
        "Status":   ["Critical" if p < threshold * 0.5 else
                     "Warning"  if p < threshold        else "Healthy"
                     for p in preds],
    }).sort_values("Pred RUL")

    table = dash_table.DataTable(
        data=pri_df.head(20).to_dict("records"),
        columns=[{"name": col, "id": col} for col in pri_df.columns],
        style_table={"overflowX": "auto", "maxHeight": "260px", "overflowY": "auto"},
        style_header={"backgroundColor": CLR["bg"], "color": CLR["accent"],
                       "fontWeight": "600", "border": f"1px solid {CLR['border']}"},
        style_cell={"backgroundColor": CLR["panel"], "color": CLR["text"],
                     "border": f"1px solid {CLR['border']}", "fontSize": "13px",
                     "padding": "6px 12px"},
        style_data_conditional=[
            {"if": {"filter_query": '{Status} = "Critical"'},
             "backgroundColor": "#2a0a0a", "color": "#ff8a8a"},
            {"if": {"filter_query": '{Status} = "Warning"'},
             "backgroundColor": "#2a1800", "color": "#ffc07a"},
            {"if": {"filter_query": '{Status} = "Healthy"'},
             "backgroundColor": "#0a2a14", "color": "#7ff4a4"},
        ],
        sort_action="native",
    )

    return html.Div([
        kpi_row,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_h), width=6),
                 dbc.Col(dcc.Graph(figure=fig_p), width=6)], className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_bars), width=12)], className="mb-3"),
        dbc.Row([dbc.Col([
            html.Div("Maintenance Priority Table (Top 20)",
                     style={"color": CLR["accent"], "fontWeight": "600",
                            "marginBottom": "8px", "fontSize": "0.9rem"}),
            table,
        ])]),
    ])


def engine_layout(c, subset, engine_id):
    train_df  = c["train_df"]
    feat_cols = c["feat_cols"]
    model     = c["model"]
    scaler    = c["scaler"]

    eng_df = train_df[train_df["unit_id"] == engine_id].sort_values("cycle").reset_index(drop=True)
    if len(eng_df) == 0:
        return html.Div("Engine not found.")

    pred_arr = rolling_rul(eng_df, feat_cols, model, scaler)
    true_arr = eng_df["rul"].values
    cycles   = eng_df["cycle"].values
    last_p   = float(pred_arr[~np.isnan(pred_arr)][-1])
    last_t   = float(true_arr[-1])

    kpi_row = dbc.Row([
        dbc.Col(kpi_card(f"{int(last_p)}", "Predicted RUL"),                         width=3),
        dbc.Col(kpi_card(f"{int(last_t)}", "True RUL"),                               width=3),
        dbc.Col(kpi_card(f"{last_p - last_t:+.1f}",  "Error",
                         CLR["red"] if abs(last_p - last_t) > 20 else CLR["green"]),  width=3),
        dbc.Col(kpi_card(int(cycles[-1]), "Cycles Observed"),                         width=3),
    ], className="mb-3")

    # RUL timeline
    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(x=cycles, y=true_arr, name="True RUL",
                                  line=dict(color=CLR["green"], width=2.5), mode="lines"))
    fig_rul.add_trace(go.Scatter(x=cycles, y=pred_arr, name="Predicted RUL",
                                  line=dict(color=CLR["accent"], width=2.5, dash="dash"), mode="lines"))
    fig_rul.add_hrect(y0=0, y1=20, fillcolor=CLR["red"],   opacity=0.12,
                      annotation_text="Critical", annotation_font_color=CLR["red"])
    fig_rul.add_hrect(y0=20, y1=50, fillcolor=CLR["amber"], opacity=0.08,
                      annotation_text="Warning",  annotation_font_color=CLR["amber"])
    fig_rul.update_layout(**pb(
        title=f"Engine {engine_id} — RUL Timeline",
        xaxis_title="Cycle", yaxis_title="RUL",
        height=300, margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor=CLR["border"]), yaxis=dict(gridcolor=CLR["border"]),
    ))

    # Sensor subplots
    top_feats = feat_cols[:9]
    fig_s = make_subplots(rows=3, cols=3, shared_xaxes=True,
                          subplot_titles=top_feats,
                          vertical_spacing=0.08, horizontal_spacing=0.07)
    pal = [CLR["accent"], CLR["amber"], CLR["green"], CLR["purple"],
           CLR["red"], CLR["teal"], "#f4a261", "#a78bfa", "#4da6ff"]
    for i, col in enumerate(top_feats):
        r_, c_ = divmod(i, 3)
        fig_s.add_trace(go.Scatter(
            x=cycles, y=eng_df[col], mode="lines",
            line=dict(color=pal[i], width=1.8), showlegend=False,
        ), row=r_ + 1, col=c_ + 1)
    fig_s.update_layout(**pb(
        title=f"Engine {engine_id} — Sensor Profiles",
        height=560, margin=dict(l=50, r=20, t=60, b=40),
    ))
    for ax in fig_s.layout:
        if ax.startswith(("xaxis", "yaxis")):
            fig_s.layout[ax].update(gridcolor=CLR["border"])

    # SHAP
    feats_arr = eng_df[feat_cols].values.astype(np.float32)
    n = len(feats_arr)
    w = feats_arr[-WINDOW:] if n >= WINDOW else np.vstack(
        [np.tile(feats_arr[0], (WINDOW - n, 1)), feats_arr])
    t_n = np.arange(WINDOW, dtype=np.float32); t_n -= t_n.mean()
    denom = float((t_n ** 2).sum())
    mu_ = w.mean(0); si_ = w.std(0)
    sl_ = (t_n[:, None] * (w - mu_)).sum(0) / denom
    row_raw = np.empty(4 * len(feat_cols), dtype=np.float32)
    row_raw[0::4] = w[-1]; row_raw[1::4] = mu_; row_raw[2::4] = si_; row_raw[3::4] = sl_
    row_sc = scaler.transform(row_raw.reshape(1, -1))

    try:
        import shap
        expl = shap.TreeExplainer(model)
        sv   = expl.shap_values(row_sc)[0]
        wfn  = c["win_feat_names"]
        shap_df = (pd.DataFrame({"feature": wfn, "shap": sv})
                     .pipe(lambda df: df.reindex(df["shap"].abs().sort_values(ascending=False).index))
                     .head(15).sort_values("shap"))
        fig_sh = go.Figure(go.Bar(
            x=shap_df["shap"], y=shap_df["feature"], orientation="h",
            marker_color=[CLR["green"] if v > 0 else CLR["red"] for v in shap_df["shap"]],
            text=[f"{v:+.2f}" for v in shap_df["shap"]], textposition="outside",
        ))
        fig_sh.add_vline(x=0, line_color=CLR["text"], line_width=1)
        fig_sh.update_layout(**pb(
            title=f"SHAP — Engine {engine_id}",
            xaxis_title="SHAP value",
            height=420, margin=dict(l=160, r=50, t=50, b=40),
            xaxis=dict(gridcolor=CLR["border"]), yaxis=dict(gridcolor=CLR["border"]),
        ))
        shap_section = dcc.Graph(figure=fig_sh)
    except Exception:
        shap_section = html.Div("SHAP unavailable (shap not installed).",
                                style={"color": CLR["muted"]})

    return html.Div([
        kpi_row,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_rul), width=12)], className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_s),   width=12)], className="mb-3"),
        dbc.Row([dbc.Col(shap_section,              width=12)]),
    ])


def naval_layout():
    preds_kmc = _nav_models["kMc"].predict(_nav_X_te_sc)
    preds_kmt = _nav_models["kMt"].predict(_nav_X_te_sc)
    y_kmc = _nav_y[_te_i, 0]
    y_kmt = _nav_y[_te_i, 1]

    def _parity(y_true, y_pred, name, color):
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                                  marker=dict(color=color, opacity=0.4, size=4), name=name))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                  line=dict(color=CLR["red"], dash="dash"), name="Perfect"))
        fig.update_layout(**pb(
            title=f"{name} — Predicted vs Actual",
            xaxis_title="True", yaxis_title="Predicted",
            height=340, margin=dict(l=50, r=20, t=50, b=40),
            xaxis=dict(gridcolor=CLR["border"]), yaxis=dict(gridcolor=CLR["border"]),
        ))
        return fig

    fig_kmc = _parity(y_kmc, preds_kmc, "kMc (Compressor)", CLR["accent"])
    fig_kmt = _parity(y_kmt, preds_kmt, "kMt (Turbine)",    CLR["purple"])

    # Feature importance
    fi_kmc = _nav_models["kMc"].feature_importances_
    fi_kmt = _nav_models["kMt"].feature_importances_
    fi_df = pd.DataFrame({"Feature": NAVAL_FEATURE_COLS, "kMc": fi_kmc, "kMt": fi_kmt}
                         ).sort_values("kMc", ascending=False)
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(x=fi_df["kMc"], y=fi_df["Feature"], orientation="h",
                             name="kMc", marker_color=CLR["accent"], opacity=0.85))
    fig_fi.add_trace(go.Bar(x=fi_df["kMt"], y=fi_df["Feature"], orientation="h",
                             name="kMt", marker_color=CLR["purple"], opacity=0.85))
    fig_fi.update_layout(**pb(
        barmode="group", title="LightGBM Feature Importance",
        xaxis_title="Importance",
        height=420, margin=dict(l=210, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor=CLR["border"]),
        yaxis=dict(gridcolor=CLR["border"], autorange="reversed"),
    ))

    # Operating condition scatter
    lever = _nav_X_df["lever_position"].values[_te_i]
    fig_op = px.scatter(x=lever, y=preds_kmc, color=y_kmc,
                         color_continuous_scale="Plasma",
                         labels={"x": "Lever Position", "y": "Predicted kMc", "color": "True kMc"},
                         title="kMc Predictions across Operating Conditions")
    fig_op.update_layout(**pb(
        height=320, margin=dict(l=50, r=20, t=50, b=40),
    ))
    fig_op.update_traces(marker=dict(size=4, opacity=0.6))

    kpi_row = dbc.Row([
        dbc.Col(kpi_card(f"{rmse(y_kmc, preds_kmc):.5f}", "kMc RMSE", CLR["accent"]), width=2),
        dbc.Col(kpi_card(f"{r2(y_kmc,  preds_kmc):.5f}",  "kMc R²",   CLR["green"]),  width=2),
        dbc.Col(kpi_card(f"{rmse(y_kmt, preds_kmt):.5f}", "kMt RMSE", CLR["purple"]), width=2),
        dbc.Col(kpi_card(f"{r2(y_kmt,  preds_kmt):.5f}",  "kMt R²",   CLR["teal"]),   width=2),
        dbc.Col(kpi_card(f"{len(_nav_X_df):,}",            "Op. Points"),              width=2),
        dbc.Col(kpi_card(f"{len(NAVAL_FEATURE_COLS)}",     "Features"),                width=2),
    ], className="mb-3")

    return html.Div([
        kpi_row,
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_kmc), width=6),
                 dbc.Col(dcc.Graph(figure=fig_kmt), width=6)], className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_op),  width=6),
                 dbc.Col(dcc.Graph(figure=fig_fi),  width=6)]),
    ])


def report_layout():
    rows = []
    for s in SUBSETS:
        m = _cache[s]["metrics"]
        rows.append({"Subset": s,
                     "Condition": "1 op" if s in ("FD001", "FD003") else "6 ops",
                     "Fault": "1 type" if s in ("FD001", "FD002") else "2 types",
                     "RMSE": m["RMSE"], "MAE": m["MAE"],
                     "R²": m["R²"], "NASA Score": m["NASA Score"]})
    cmapss_df = pd.DataFrame(rows)

    nav_rows = [
        {"Model": "LightGBM (kMc)", "RMSE": _nav_metrics["RMSE_kMc"],
         "MAE": _nav_metrics["MAE_kMc"],  "R²": _nav_metrics["R²_kMc"]},
        {"Model": "LightGBM (kMt)", "RMSE": _nav_metrics["RMSE_kMt"],
         "MAE": _nav_metrics["MAE_kMt"],  "R²": _nav_metrics["R²_kMt"]},
    ]
    naval_df = pd.DataFrame(nav_rows)

    def _tbl(df):
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": CLR["bg"], "color": CLR["accent"],
                           "fontWeight": "600", "border": f"1px solid {CLR['border']}"},
            style_cell={"backgroundColor": CLR["panel"], "color": CLR["text"],
                         "border": f"1px solid {CLR['border']}", "fontSize": "13px",
                         "padding": "7px 14px"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": CLR["bg"]}
            ],
        )

    # Radar chart
    cats  = ["RMSE (inv)", "MAE (inv)", "R²", "NASA (inv)"]
    radar = go.Figure()
    for s in SUBSETS:
        m = _cache[s]["metrics"]
        vals = [
            max(0, 1 - m["RMSE"] / 50),
            max(0, 1 - m["MAE"]  / 40),
            max(0, m["R²"]),
            max(0, 1 - min(1, m["NASA Score"] / 5000)),
        ]
        radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name=s, opacity=0.7,
        ))
    radar.update_layout(**pb(
        title="Performance Radar — CMAPSS Subsets",
        height=420, margin=dict(l=50, r=50, t=60, b=50),
        polar=dict(
            bgcolor=CLR["panel"],
            radialaxis=dict(visible=True, color=CLR["muted"], gridcolor=CLR["border"]),
            angularaxis=dict(color=CLR["text"], gridcolor=CLR["border"]),
        ),
        legend=dict(orientation="h"),
    ))

    findings = html.Ul([
        html.Li("Window-XGBoost generalises across all four CMAPSS subsets; lowest RMSE on FD001 "
                "(single operating condition, single fault mode).",
                style={"marginBottom": "8px"}),
        html.Li("Multi-condition subsets FD002/FD004 show higher errors — operating variability "
                "compresses the sensor signal-to-noise ratio.",
                style={"marginBottom": "8px"}),
        html.Li("The 30-cycle temporal window captures degradation trajectories that point-in-time "
                "models miss, yielding significant R² improvement over Ridge and tabular XGBoost.",
                style={"marginBottom": "8px"}),
        html.Li("Naval LightGBM achieves near-perfect R² on both kMc and kMt, reflecting the "
                "clean physics-based simulation behind that dataset.",
                style={"marginBottom": "8px"}),
        html.Li("The NASA asymmetric score penalises late predictions more heavily than early ones, "
                "guiding the model toward conservative (safety-first) RUL estimates.",
                style={"marginBottom": "8px"}),
    ], style={"color": CLR["text"], "fontSize": "0.88rem", "paddingLeft": "18px"})

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("CMAPSS — Window-XGBoost",
                        style={"color": CLR["accent"], "marginBottom": "10px"}),
                _tbl(cmapss_df),
            ], width=7),
            dbc.Col([
                html.H5("Naval — LightGBM",
                        style={"color": CLR["accent"], "marginBottom": "10px"}),
                _tbl(naval_df),
            ], width=5),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=radar), width=6),
            dbc.Col([
                html.H5("Key Findings",
                        style={"color": CLR["accent"], "marginBottom": "12px"}),
                findings,
            ], width=6),
        ]),
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
