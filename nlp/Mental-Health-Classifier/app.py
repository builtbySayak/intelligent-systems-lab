import os
import joblib
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import cohere
except ImportError:
    cohere = None

#SAFE cohere API key fetch
CO_API_KEY = None
if "COHERE_API_KEY" in os.environ:
    CO_API_KEY = os.environ["COHERE_API_KEY"]
elif hasattr(st, "secrets") and "COHERE_API_KEY" in st.secrets:
    CO_API_KEY = st.secrets["COHERE_API_KEY"]

CO_MODEL = "command-r-plus"
co_cli = cohere.Client(CO_API_KEY) if (cohere and CO_API_KEY) else None
#UI setup
st.set_page_config(page_title="Mental-Health Classifier", layout="centered")
st.title("Mental-Health Classifier — MSc 2025")

#Labels / colors
LABELS = ["neutral", "academic_stress", "relationship_issues",
          "existential_crisis", "social_isolation"]

INT     = {"neutral":0, "social_isolation":.25, "relationship_issues":.5,
           "academic_stress":.75, "existential_crisis":1}
INT_COL = ["#4CAF50", "#9CCC65", "#FFB74D", "#FF7043", "#D32F2F"]

def colour(v): return INT_COL[min(int(v*4), 4)]

#Locate model root
CANDIDATE_ROOTS = [
    "/Users/sayak/Desktop/MSc_Project_Sayak/Code/Pretrained-Models",
    "/Users/sayak/Desktop/MSc_Project_Sayak/Code/Models",
]

def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

BASE = first_existing(CANDIDATE_ROOTS)
if not BASE:
    st.error("Could not find a models folder. Set BASE to your "
             "`Pretrained-Models` or `Models` directory and rerun.")
    st.stop()

#Standard expected layout (adjust names if yours differ)
PATHS = {
    "LR": {
        "model": os.path.join(BASE, "LR", "emotion_classifier.joblib"),
        "vec":   os.path.join(BASE, "LR", "tfidf_vectorizer.joblib"),
    },
    "SVM": {
        "model": os.path.join(BASE, "SVM", "emotion_svm_classifier.joblib"),
        "vec":   os.path.join(BASE, "SVM", "tfidf_vectorizer.joblib"),
    },
    "XGB": {
        "model": os.path.join(BASE, "XGboost", "emotion_xgb_classifier.joblib"),
        "vec":   os.path.join(BASE, "XGboost", "xgb_vectorizer.joblib"),
    },
    "ROBERTA_DIR": os.path.join(BASE, "Roberta_Pretrained"),
    "STACK_META":  os.path.join(BASE, "Stacked", "meta_classifier.joblib"),
}

def _ensure_exists(path, kind="file"):
    if kind == "file" and not os.path.isfile(path):
        st.error(f"Missing file: `{path}`")
        st.stop()
    if kind == "dir" and not os.path.isdir(path):
        st.error(f"Missing directory: `{path}`")
        st.stop()

#Check what we need (others are checked lazily when used)
for m in ("LR","SVM","XGB"):
    _ensure_exists(PATHS[m]["model"])
    _ensure_exists(PATHS[m]["vec"])
_ensure_exists(PATHS["ROBERTA_DIR"], kind="dir")
_ensure_exists(PATHS["STACK_META"])

#Cached loaders
@st.cache_resource(show_spinner=False)
def load_vec_clf(model_p, vec_p):
    return joblib.load(model_p), joblib.load(vec_p)

@st.cache_resource(show_spinner=False)
def load_roberta(dir_path):
    tok = AutoTokenizer.from_pretrained(dir_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        dir_path, num_labels=len(LABELS)
    ).eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_stack_base_and_meta():
    """Load meta & all base models directly from PATHS (no base_paths.joblib)."""
    meta = joblib.load(PATHS["STACK_META"])
    base = {
        "LR":  load_vec_clf(PATHS["LR"]["model"],  PATHS["LR"]["vec"]),
        "SVM": load_vec_clf(PATHS["SVM"]["model"], PATHS["SVM"]["vec"]),
        "XGB": load_vec_clf(PATHS["XGB"]["model"], PATHS["XGB"]["vec"]),
        "ROBERTA": load_roberta(PATHS["ROBERTA_DIR"]),
    }
    return meta, base

#Predict helpers
def reorder(probs, cls):
    """Map estimator's class order to our LABELS order."""
    index = { (c if isinstance(c,str) else LABELS[int(c)]) : i
              for i, c in enumerate(cls) }
    return np.array([probs[index[l]] for l in LABELS])

def clf_predict(text, clf, vec):
    probs = reorder(
        clf.predict_proba(vec.transform([text]))[0],
        clf.classes_
    )
    return LABELS[int(np.argmax(probs))], probs

def rob_predict(text, tok, mdl):
    with torch.no_grad():
        logits = mdl(**tok(text, return_tensors="pt",
                           truncation=True, padding=True,
                           max_length=128)).logits
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return LABELS[int(np.argmax(probs))], probs

def stack_predict(text, meta, base):
    #concatenate base probabilities: LR | SVM | XGB | ROBERTA
    feats = np.hstack([
        clf_predict(text, *base["LR"])[1],
        clf_predict(text, *base["SVM"])[1],
        clf_predict(text, *base["XGB"])[1],
        rob_predict(text, *base["ROBERTA"])[1],
    ]).reshape(1, -1)
    probs = meta.predict_proba(feats)[0]
    return LABELS[int(np.argmax(probs))], probs

def tfidf_tokens(text, vec, k=20):
    row = vec.transform([text])
    inv = {v:k for k,v in vec.vocabulary_.items()}
    tok = {inv[i]:v for i,v in zip(row.indices,row.data) if len(inv[i])>2}
    return dict(sorted(tok.items(), key=lambda x:x[1], reverse=True)[:k])

def cohere_suggest(post, category):
    if not co_cli:
        return "Set COHERE_API_KEY to enable live suggestions."
    system_prompt = ("I am a UK-university wellbeing assistant "
                     "who gives short and practical advice.")
    user_prompt = (f"A student is experiencing {category.replace('_',' ')}.\n\n"
                   f"Post:\n{post.strip()}\n\n"
                   "Suggest an actionable coping strategy in 3-4 sentences.")
    try:
        resp = co_cli.chat(
            model=CO_MODEL,
            temperature=0.7,
            max_tokens=120,
            message=user_prompt,
            preamble=system_prompt,
        )
        return resp.text.strip()
    except Exception as e:
        return f"Cohere error: {e}"

#UI
UI2KEY = {
    "Logistic Regression": "LR",
    "Support Vector Machine": "SVM",
    "XG-Boost": "XGB",
    "RoBERTa": "ROBERTA",
    "Stack": "STACK",
}

text_in  = st.text_area("Paste a Reddit-style post:",
                        "I feel like I can't do this any more.", height=120)
model_ui = st.selectbox("Choose a model:", list(UI2KEY.keys()))

if st.button("Predict") and text_in.strip():
    model_key = UI2KEY[model_ui]

    try:
        if model_key in {"LR","SVM","XGB"}:
            clf, vec = load_vec_clf(PATHS[model_key]["model"], PATHS[model_key]["vec"])
            label, probs = clf_predict(text_in, clf, vec)
            wc_src = tfidf_tokens(text_in, vec) if label != "neutral" else {}

        elif model_key == "ROBERTA":
            tok, mdl = load_roberta(PATHS["ROBERTA_DIR"])
            label, probs = rob_predict(text_in, tok, mdl)
            wc_src = {}

        else:  #STACK
            meta, base = load_stack_base_and_meta()
            label, probs = stack_predict(text_in, meta, base)
            wc_src = {}

    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    distress = (label != "neutral")
    severity = INT[label] if distress else 0.0

    st.markdown(f"### **{label}**")
    st.markdown(f"**Distress?** {'Yes' if distress else 'No'}  |  "
                f"**Severity:** `{severity:.2f}`")
    st.markdown(
        f"<div style='height:20px;border-radius:4px;background:{colour(severity)}'></div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(5, 2.3))
    sns.barplot(x=probs, y=LABELS, palette="crest", ax=ax)
    ax.set_xlim(0, 1); ax.set_xlabel(""); ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)

    if distress and wc_src:
        wc = WordCloud(width=400, height=200, background_color="white",
                       colormap="autumn").generate_from_frequencies(wc_src)
        st.markdown("#### Key distress terms")
        st.image(wc.to_array())

    if distress:
        st.markdown("#### Suggestion")
        st.success(cohere_suggest(text_in, label))

st.caption("MSc-Project 2025 School of EECS - QMUL")
