# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# -------------------------------------------------------------
# Page config & Theming
# -------------------------------------------------------------
st.set_page_config(
    page_title="ImmoLab · Pricing & Typology",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND = {
    "bg": "#0f172a",       # slate-900
    "card": "#111827",    # gray-900
    "accent": "#a78bfa",  # violet-400
    "accent2": "#fb923c", # orange-400
    "muted": "#94a3b8",   # slate-400
    "text": "#e5e7eb",    # gray-200
}

st.markdown(
    f"""
    <style>
      .main .block-container {{ padding-top: 1.2rem; max-width: 1400px; }}
      .im-card {{
        background: linear-gradient(135deg, {BRAND['card']} 0%, #0b1220 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 16px 18px; color: {BRAND['text']};
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
      }}
      .im-kpi {{ font-weight: 800; font-size: 30px; color: {BRAND['accent']}; }}
      .im-sub {{ color: {BRAND['muted']}; font-size: 13px; }}
      .im-tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px;
                 background: {BRAND['accent2']}; color: #0b1220; font-weight: 700;
                 font-size: 11px; margin-left: 8px; }}
      .im-pill {{ display:inline-block; margin-top:10px; padding:6px 10px; border-radius:999px;
                  font-size:12px; font-weight:700; letter-spacing:0.2px; border:1px solid rgba(255,255,255,0.12); }}
      .pill-reg {{ background: rgba(167,139,250,0.10); color:#a78bfa; border-color: rgba(167,139,250,0.35); }}
      .pill-clf {{ background: rgba(16,185,129,0.10); color:#10b981; border-color: rgba(16,185,129,0.35); }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Load artifacts (exported by the notebook)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    with open("models/rf_reg.pkl", "rb") as f:
        rf_reg = pickle.load(f)
    with open("models/rf_clf.pkl", "rb") as f:
        rf_clf = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    return rf_reg, rf_clf, enc

@st.cache_data(show_spinner=False)
def try_load_test():
    if os.path.exists("test.csv"):
        df = pd.read_csv("test.csv")
        for c in ["TotalBsmtSF", "GarageCars", "GarageArea"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)
                if c == "GarageCars":
                    df[c] = df[c].astype(int)
        return df
    return None

rf_reg, rf_clf, enc = load_models()
REG_FEATURES = enc["REG_FEATURES"]
CLF_FEATURES = enc["CLF_FEATURES"]
TARGET_REG = enc["TARGET_REG"]
TARGET_CLF = enc["TARGET_CLF"]
label_encoders_reg = enc["label_encoders_reg"]
label_encoders_clf = enc["label_encoders_clf"]
le_target = enc["le_target"]

TEST_DF = try_load_test()

BLDG_LABELS = {
    "1Fam": "Maison individuelle",
    "2fmCon": "Bi-familiale",
    "Duplex": "Duplex",
    "Twnhs": "Maison de ville",
    "TwnhsE": "Maison de ville (extrémité)",
}

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def safe_encode(value, encoder):
    classes = list(getattr(encoder, "classes_", []))
    if classes and value in classes:
        return int(np.where(np.array(classes) == value)[0][0])
    st.warning(f"Valeur inconnue détectée ('{value}') — remplacement par '{classes[0] if classes else '0'}'.")
    return 0


def build_input_form(default_row: pd.Series | None = None):
    st.markdown("### 🎚️ Paramétrage de l'observation")
    colA, colB, colC = st.columns(3)

    # Possible classes
    nbhd_classes = []
    if "Neighborhood" in label_encoders_reg and hasattr(label_encoders_reg["Neighborhood"], "classes_"):
        nbhd_classes = list(label_encoders_reg["Neighborhood"].classes_)
    elif "Neighborhood" in label_encoders_clf and hasattr(label_encoders_clf["Neighborhood"], "classes_"):
        nbhd_classes = list(label_encoders_clf["Neighborhood"].classes_)

    hs_classes = list(label_encoders_clf["HouseStyle"].classes_) if (
        "HouseStyle" in label_encoders_clf and hasattr(label_encoders_clf["HouseStyle"], "classes_")
    ) else []

    def d(col, fallback):
        if default_row is not None and col in default_row.index:
            val = default_row[col]
            try:
                if pd.isna(val):
                    return fallback
            except Exception:
                pass
            return val
        return fallback

    with colA:
        GrLivArea = st.number_input("Surface habitable (GrLivArea)", 0, 10000, int(d("GrLivArea", 1500)), 10)
        TotalBsmtSF = st.number_input("Surface sous-sol (TotalBsmtSF)", 0, 4000, int(d("TotalBsmtSF", 800)), 10)
        OverallQual = st.slider("Qualité générale (OverallQual)", 1, 10, int(d("OverallQual", 6)))
        OverallCond = st.slider("État général (OverallCond)", 1, 10, int(d("OverallCond", 6)))

    with colB:
        YearBuilt = st.number_input("Année de construction (YearBuilt)", 1870, 2020, int(d("YearBuilt", 1978)), 1)
        YearRemodAdd = st.number_input("Année de rénovation (YearRemodAdd)", 1950, 2020, int(d("YearRemodAdd", 2005)), 1)
        GarageCars = st.number_input("Places garage (GarageCars)", 0, 6, int(d("GarageCars", 1)), 1)
        GarageArea = st.number_input("Surface garage (GarageArea)", 0, 2000, int(d("GarageArea", 400)), 10)

    with colC:
        LotArea = st.number_input("Surface terrain (LotArea)", 0, 200000, int(d("LotArea", 8000)), 50)
        PoolArea = st.number_input("Surface piscine (PoolArea)", 0, 2000, int(d("PoolArea", 0)), 10)
        Fireplaces = st.number_input("Cheminées (Fireplaces)", 0, 5, int(d("Fireplaces", 1)), 1)
        TotRmsAbvGrd = st.number_input("Pièces totales (TotRmsAbvGrd)", 1, 20, int(d("TotRmsAbvGrd", 6)), 1)

    c1, c2 = st.columns(2)
    with c1:
        Neighborhood = nbhd_classes[0] if not nbhd_classes else (
            st.selectbox("Quartier (Neighborhood)", nbhd_classes, index=(nbhd_classes.index(d("Neighborhood", nbhd_classes[0])) if d("Neighborhood", None) in nbhd_classes else 0))
        )
    with c2:
        HouseStyle = hs_classes[0] if not hs_classes else (
            st.selectbox("Style (HouseStyle)", hs_classes, index=(hs_classes.index(d("HouseStyle", hs_classes[0])) if d("HouseStyle", None) in hs_classes else 0))
        )

    reg_vals = {
        "GrLivArea": GrLivArea,
        "TotalBsmtSF": TotalBsmtSF,
        "LotArea": LotArea,
        "BedroomAbvGr": int(d("BedroomAbvGr", 3)),
        "FullBath": int(d("FullBath", 2)),
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "OverallQual": OverallQual,
        "OverallCond": OverallCond,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "Neighborhood": Neighborhood,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,
        "PoolArea": PoolArea,
        "Fireplaces": Fireplaces,
    }

    clf_vals = {
        "GrLivArea": GrLivArea,
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "OverallQual": OverallQual,
        "YearBuilt": YearBuilt,
        "GarageCars": GarageCars,
        "Neighborhood": Neighborhood,
        "HouseStyle": HouseStyle,
    }
    return reg_vals, clf_vals


def encode_for_regression(values: dict) -> np.ndarray:
    row = []
    for col in REG_FEATURES:
        val = values.get(col, 0)
        if col in label_encoders_reg:
            val = safe_encode(val, label_encoders_reg[col])
        row.append(float(val))
    return np.array(row, dtype=float).reshape(1, -1)


def encode_for_classification(values: dict) -> np.ndarray:
    row = []
    for col in CLF_FEATURES:
        val = values.get(col, 0)
        if col in label_encoders_clf:
            val = safe_encode(val, label_encoders_clf[col])
        row.append(float(val))
    return np.array(row, dtype=float).reshape(1, -1)


def predict_all(reg_vals: dict, clf_vals: dict):
    X_reg = encode_for_regression(reg_vals)
    X_clf = encode_for_classification(clf_vals)

    price_pred = float(rf_reg.predict(X_reg)[0])
    tree_preds = np.array([t.predict(X_reg)[0] for t in rf_reg.estimators_])
    price_std = float(np.std(tree_preds))
    ci_low = max(0.0, price_pred - 1.96 * price_std)
    ci_high = price_pred + 1.96 * price_std

    y_code = int(rf_clf.predict(X_clf)[0])
    y_label = le_target.inverse_transform([y_code])[0]
    y_label_h = BLDG_LABELS.get(y_label, y_label)

    if hasattr(rf_clf, "predict_proba"):
        proba = rf_clf.predict_proba(X_clf)[0]
        classes = list(le_target.classes_)
    else:
        proba = np.ones(len(le_target.classes_)) / len(le_target.classes_)
        classes = list(le_target.classes_)

    return price_pred, (ci_low, ci_high), y_label, y_label_h, proba, classes, tree_preds

# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class=\"im-card\">
          <h3>🏗️ ImmoLab</h3>
          <div class=\"im-sub\">Ames · Pricing & Typology</div>
          <div style=\"margin-top:6px;\">
            <span class=\"im-tag\">Random Forest</span>
            <span class=\"im-tag\">15 features</span>
            <span class=\"im-tag\">7 features (clf)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    TEST_MODE = st.radio("Source des données", ["ID depuis test.csv", "Saisie manuelle"], index=0 if TEST_DF is not None else 1)

    selected_row = None
    if TEST_MODE == "ID depuis test.csv" and TEST_DF is not None:
        selected_id = st.selectbox("Choisir un ID", TEST_DF["Id"].tolist())
        selected_row = TEST_DF.loc[TEST_DF["Id"] == selected_id].iloc[0]
    elif TEST_DF is None:
        st.info("Aucun test.csv détecté. Bascule vers la saisie manuelle.")

    # Models & features card
    st.markdown(
        f"""
        <div class=\"im-card\" style=\"margin-top:14px;\"> 
          <div style=\"font-weight:700;\">Modèles</div>
          <div class=\"im-sub\">Random Forest Regressor<br>Random Forest Classifier</div>
          <div style=\"height:8px;\"></div>
          <div style=\"font-weight:700;\">Variables</div>
          <div class=\"im-sub\">Régression : {len(REG_FEATURES)} features<br>Classification : {len(CLF_FEATURES)} features</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.markdown(
    f"""
    <div class=\"im-card\">
      <h2 style=\"margin:0;\">🔮 ESTIMATION ET TOPOLOGIE DU BIEN</h2>
      <div class=\"im-sub\">Modèles issus du notebook · Random Forect (régression & classification)</div>
    </div>
    """,
    unsafe_allow_html=True,
)
# --- Description d'accueil (brève) ---
st.markdown(
    
    """
        **Bienvenue sur ImmoLab.**  
    Choisissez un bien (ou saisissez ses caractéristiques) puis cliquez sur **« Lancer la prédiction »** pour obtenir :
    - le **prix estimé** (+ intervalle 95 %),
    - le **type de bâtiment** (+ probabilités),
    - et une **analyse des prédictions**.
    
    Vous pouvez aussi **charger un CSV** dans l’onglet *Scoring par lot*.
    """
)

# -------------------------------------------------------------
# Predictions Tab
# -------------------------------------------------------------
pred_tab, explain_tab, batch_tab = st.tabs(["🧮 Prédictions", "🔍 Explicabilité", "📦 Scoring par lot"])


with pred_tab:
    reg_vals, clf_vals = build_input_form(selected_row)
    run = st.button("Lancer la prédiction", type="primary")

    if run:
        price_pred, (ci_low, ci_high), y_label, y_label_h, proba, classes, tree_preds = predict_all(reg_vals, clf_vals)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class=\"im-card\">
                  <div class=\"im-sub\">Estimation de prix (USD)</div>
                  <div class=\"im-kpi\">${price_pred:,.0f}</div>
                  <div class=\"im-sub\">Intervalle 95 %: ${ci_low:,.0f} → ${ci_high:,.0f}</div>
                  <div class=\"im-pill pill-reg\">RF Regressor · R² ≈ 0.88</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class=\"im-card\">
                  <div class=\"im-sub\">Type de bâtiment</div>
                  <div class=\"im-kpi\">{y_label_h} <span class=\"im-sub\">({y_label})</span></div>
                  <div class=\"im-pill pill-clf\">RF Classifier · Acc. ≈ 0.89</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # === Analyse des prédictions ===
        st.markdown("#### Analyse des prédictions")
        g1, g2 = st.columns(2)
        with g1:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=tree_preds,
                nbinsx=30,
                marker_color="#7286D3",
                marker_line_color="#4B5FA5",
                marker_line_width=0.4,
                opacity=0.9,
            ))
            fig_dist.add_vline(
                x=price_pred,
                line_dash="dot",
                line_color="#ef4444",
                line_width=2,
                annotation_text=f" ${price_pred:,.0f}",
                annotation_font_color="#ef4444",
                annotation_font_size=12,
                annotation_position="top right",
            )
            fig_dist.update_layout(
                height=360,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=30, b=10),
                title_text="Distribution des 200 arbres — Prix ($)",
                xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
                yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title="Arbres"),
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with g2:
            order = np.argsort(proba)[::-1]
            classes_sorted = [classes[i] for i in order]
            labels_sorted = [f"{BLDG_LABELS.get(c, c)} ({c})" for c in classes_sorted]
            values_sorted = [proba[i] for i in order]
            colors = [BRAND['accent2'] if c == y_label else BRAND['muted'] for c in classes_sorted]

            fig_prob = go.Figure(go.Bar(
                x=values_sorted,
                y=labels_sorted,
                orientation="h",
                marker_color=colors,
                text=[f"{v*100:.1f}%" for v in values_sorted],
                textposition="outside",
            ))
            fig_prob.update_layout(
                height=360,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(range=[0, 1.15], tickformat='.0%'),
                title_text="Probabilités par type de bâtiment",
                showlegend=False,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

# -------------------------------------------------------------
# Explainability Tab
# -------------------------------------------------------------
with explain_tab:
    st.markdown("#### Importance des variables")
    col1, col2 = st.columns(2)

    with col1:
        imp_reg = pd.Series(rf_reg.feature_importances_, index=REG_FEATURES).sort_values()
        fig1 = go.Figure(go.Bar(x=imp_reg.values * 100, y=imp_reg.index, orientation='h', marker_color=BRAND['accent']))
        fig1.update_layout(title_text="Régression · contribution (%)", xaxis_title="% (somme = 100)", height=420, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        imp_clf = pd.Series(rf_clf.feature_importances_, index=CLF_FEATURES).sort_values()
        fig2 = go.Figure(go.Bar(x=imp_clf.values * 100, y=imp_clf.index, orientation='h', marker_color=BRAND['accent2']))
        fig2.update_layout(title_text="Classification · contribution (%)", xaxis_title="% (somme = 100)", height=420, plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.caption("Note : importances basées sur la réduction d'impureté. Compléter par permutation importance/SHAP si nécessaire.")

# -------------------------------------------------------------
# Batch Tab
# -------------------------------------------------------------
with batch_tab:
    st.markdown("#### Charger un CSV et obtenir des prédictions")
    up = st.file_uploader("Déposez un CSV contenant REG_FEATURES / CLF_FEATURES", type=["csv"]) 

    if up is not None:
        df_in = pd.read_csv(up)

        def encode_row_reg(row):
            vals = {c: row.get(c, np.nan) for c in REG_FEATURES}
            return encode_for_regression(vals)

        def encode_row_clf(row):
            vals = {c: row.get(c, np.nan) for c in CLF_FEATURES}
            return encode_for_classification(vals)

        Xr = np.vstack([encode_row_reg(row).ravel() for _, row in df_in.iterrows()])
        Xc = np.vstack([encode_row_clf(row).ravel() for _, row in df_in.iterrows()])

        y_price = rf_reg.predict(Xr)
        y_type_code = rf_clf.predict(Xc)
        y_type = le_target.inverse_transform(y_type_code)

        out = df_in.copy()
        out["Pred_SalePrice"] = y_price
        out["Pred_BldgType"] = y_type

        st.success(f"Prédictions générées pour {len(out):,} lignes.")
        st.dataframe(out.head(20))

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Télécharger les prédictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

st.caption("Interface Streamlit adaptée au notebook · artefacts chargés depuis ./models/*")
