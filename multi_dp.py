import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Disease Predictor — Clean UI", layout="centered")

MODEL_FILES = {
    "Liver": {"pkl": "liver_xgb_pipeline.pkl", "feats": "liver_features.pkl"},
    "Kidney": {"pkl": "kidney_xgb_pipeline.pkl", "feats": "kidney_features.pkl"},
    "Parkinson": {"pkl": "parkinson_xgb_pipeline.pkl", "feats": "parkinson_features.pkl"},
}

# ---------- Helpers ----------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_pipeline_and_features(disease):
    files = MODEL_FILES[disease]
    pkl_path, feat_path = files["pkl"], files["feats"]
    if not os.path.exists(pkl_path):
        st.error(f"Missing model file: {pkl_path}")
        return None, None
    pipeline = load_pickle(pkl_path)
    feat_list = None
    if os.path.exists(feat_path):
        feat_list = load_pickle(feat_path)
    return pipeline, feat_list

def infer_features_from_pipeline(pipeline):
    try:
        preproc = None
        # try common names first
        for name in ("preproc","preprocessor","preproc_","preprocessor_"):
            preproc = getattr(pipeline.named_steps.get(name), None) if pipeline is not None else None
            if preproc is not None:
                break
        # fallback: find first transformer with transformers_
        if preproc is None:
            for nm, obj in pipeline.named_steps.items():
                if hasattr(obj, "transformers_"):
                    preproc = obj
                    break
        if preproc is None:
            return None
        transformers = getattr(preproc, "transformers_", [])
        cols = []
        for name, trans, colnames in transformers:
            if colnames is None:
                continue
            cols.extend(list(colnames))
        return cols if cols else None
    except Exception:
        return None

def build_input_widgets(feat_list, ohe_map):
    """Return dict of feature->value built from widgets."""
    data = {}
    for feat in feat_list:
        label = feat.replace("_", " ").title()
        if feat in ohe_map and ohe_map[feat]:
            opts = ["(missing)"] + ohe_map[feat]
            choice = st.selectbox(label, opts, key=feat)
            data[feat] = None if choice == "(missing)" else choice
        else:
            # numeric input
            data[feat] = st.number_input(label, value=0.0, format="%.4f", key=feat)
    return data

def get_ohe_categories(pipeline):
    cats_map = {}
    try:
        preproc = pipeline.named_steps.get("preproc") or pipeline.named_steps.get("preprocessor")
        if preproc is None:
            # try find ColumnTransformer in pipeline
            for nm, obj in pipeline.named_steps.items():
                if hasattr(obj, "named_transformers_"):
                    preproc = obj
                    break
        if preproc is None:
            return cats_map
        cat_tr = getattr(preproc, "named_transformers_", {}).get("cat")
        if cat_tr is None:
            # try by index (some pipelines have num at 0, cat at 1)
            transformers = getattr(preproc, "transformers_", [])
            for name, trans, cols in transformers:
                if "cat" in name.lower():
                    try:
                        cat_tr = trans
                        break
                    except Exception:
                        pass
        if cat_tr and hasattr(cat_tr, "named_steps"):
            ohe = cat_tr.named_steps.get("onehot")
        else:
            ohe = None
        if ohe is not None and hasattr(ohe, "categories_"):
            # try to retrieve column names from transformers_ order
            transformers = getattr(preproc, "transformers_", [])
            cat_cols = None
            for name, trans, cols in transformers:
                if "cat" in name.lower():
                    cat_cols = cols
                    break
            if cat_cols:
                for col, cats in zip(cat_cols, ohe.categories_):
                    cats_map[col] = list(cats)
    except Exception:
        pass
    return cats_map

# ---------- UI ----------
st.title("🩺 Disease Predictor — Fast & Effective")
st.write("Select a disease, fill patient values, and click **Predict**. Probabilities are shown with guidance below.")

disease = st.selectbox("Choose disease", list(MODEL_FILES.keys()))

pipeline, feat_list = load_pipeline_and_features(disease)
if pipeline is None:
    st.stop()

# infer features if feature list missing
if feat_list is None:
    feat_list = infer_features_from_pipeline(pipeline)
    if feat_list is None:
        st.error("Could not determine input features automatically. Please provide the features `.pkl` file.")
        st.stop()

# detect OHE categories for nice dropdowns
ohe_map = get_ohe_categories(pipeline)

st.markdown("#### Patient input")
with st.form("main_form", clear_on_submit=False):
    input_values = build_input_widgets(feat_list, ohe_map)
    submitted = st.form_submit_button("Predict")

# neat explanation box
st.write("")
st.info("How to interpret probabilities: The app shows the model's probability that the patient *has* the condition (class 1). Increase sensitivity by lowering the decision threshold if you want to flag more possible positives.")

# result area
if submitted:
    # ensure column order
    input_df = pd.DataFrame([input_values], columns=feat_list)
    st.write("**Input preview**")
    st.dataframe(input_df.T.rename(columns={0:"value"}))

    # predict
    try:
        probs = pipeline.predict_proba(input_df)
        pred = pipeline.predict(input_df)
        # if binary
        if probs.shape[1] == 2:
            prob_pos = float(probs[0, 1])
            prob_neg = float(probs[0, 0])
            # Decision threshold slider (small, default 0.5)
            thresh = st.slider("Decision threshold for positive class", 0.0, 1.0, 0.50, 0.01, help="Lower threshold -> more positives detected (higher sensitivity).")
            label_at_thresh = int(prob_pos >= thresh)

            # Result card: color-coded by risk
            if prob_pos >= 0.75:
                card_status = ("🔴 High Risk", "This case has high predicted probability.", "danger")
            elif prob_pos >= 0.4:
                card_status = ("🟡 Medium Risk", "Moderate probability — consider follow-up.", "warning")
            else:
                card_status = ("🟢 Low Risk", "Low predicted probability.", "success")

            col1, col2 = st.columns([1,2])
            with col1:
                if card_status[2] == "danger":
                    st.error(card_status[0])
                elif card_status[2] == "warning":
                    st.warning(card_status[0])
                else:
                    st.success(card_status[0])
                st.write(card_status[1])

            with col2:
                st.metric("Model predicted class (raw)", int(pred[0]))
                st.write(f"Probability (no disease / disease): {prob_neg:.4f}  /  {prob_pos:.4f}")
                st.write(f"Predicted label at threshold {thresh:.2f}: **{label_at_thresh}**")

        else:
            # multiclass fallback
            prob_series = pd.Series(probs[0], index=[f"class_{i}" for i in range(probs.shape[1])])
            st.write("Predicted class:", int(pred[0]))
            st.dataframe(prob_series.sort_values(ascending=False).to_frame("probability"))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# footer — compact (no tuning parameters shown)
st.markdown("---")
st.caption("Notes: This app uses pre-trained XGBoost pipelines (preprocessor + model). The tuning parameters are intentionally hidden during prediction for clarity. If you want model details, enable debug mode separately.")
