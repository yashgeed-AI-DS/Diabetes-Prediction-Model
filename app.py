"""
GlucoLens – Diabetes Risk Assessment (Streamlit)

Required files:
  logistic_regression_model.pkl
  scaler.pkl

The app searches the repository recursively for these files,
so they do not have to be directly beside app.py.
"""

import time
import sys
import importlib.util
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GlucoLens – Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# DEBUG / DEPENDENCY CHECK
# ============================================================

# Temporary diagnostic to check the deployment environment.
# This helps identify the "No module named 'sklearn'" error.

SKLEARN_SPEC = importlib.util.find_spec("sklearn")

if SKLEARN_SPEC is None:
    st.warning(
        "⚠️ scikit-learn is NOT available in the current environment."
    )
else:
    st.success(
        "✅ scikit-learn is available."
    )

# You can see the exact Python version and sklearn status
# while troubleshooting the deployment.
with st.expander("🔧 Deployment Debug Information"):

    st.write("**Python version:**")
    st.code(sys.version)

    st.write("**sklearn available:**")
    st.code(str(SKLEARN_SPEC))


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

MODEL_FILENAME = "logistic_regression_model.pkl"
SCALER_FILENAME = "scaler.pkl"


def find_file(filename):
    """
    Search for a required file.

    First checks the same folder as app.py.
    Then searches all subfolders recursively.
    """

    # 1. Check directly beside app.py
    direct_path = BASE / filename

    if direct_path.is_file():
        return direct_path

    # 2. Search recursively
    matches = list(BASE.rglob(filename))

    if matches:
        return matches[0]

    return None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_artifacts():
    """
    Locate and load the trained Logistic Regression model
    and StandardScaler.

    Supports both:
    - pickle files created with pickle.dump()
    - joblib files created with joblib.dump()
    """

    model_path = find_file(MODEL_FILENAME)
    scaler_path = find_file(SCALER_FILENAME)

    # ========================================================
    # CHECK MODEL FILE
    # ========================================================

    if model_path is None:
        st.error("❌ Logistic Regression model file was not found.")

        st.write("### Files available in the application folder:")

        try:
            files = [
                str(p.relative_to(BASE))
                for p in BASE.rglob("*")
                if p.is_file()
            ]

            if files:
                st.code("\n".join(files))
            else:
                st.write("No files were found.")

        except Exception:
            pass

        st.stop()

    # ========================================================
    # CHECK SCALER FILE
    # ========================================================

    if scaler_path is None:
        st.error("❌ Scaler file was not found.")

        st.write("### Files available in the application folder:")

        try:
            files = [
                str(p.relative_to(BASE))
                for p in BASE.rglob("*")
                if p.is_file()
            ]

            if files:
                st.code("\n".join(files))
            else:
                st.write("No files were found.")

        except Exception:
            pass

        st.stop()

    # ========================================================
    # FILE INFORMATION
    # ========================================================

    with st.expander("📁 Model File Information"):

        st.write("**Model path:**")
        st.code(str(model_path))

        st.write("**Model size:**")
        st.code(f"{model_path.stat().st_size:,} bytes")

        st.write("**Scaler path:**")
        st.code(str(scaler_path))

        st.write("**Scaler size:**")
        st.code(f"{scaler_path.stat().st_size:,} bytes")

    # ========================================================
    # CHECK SCIKIT-LEARN
    # ========================================================

    if SKLEARN_SPEC is None:
        st.error(
            "❌ scikit-learn is not installed in the deployment environment."
        )

        st.info(
            "Make sure requirements.txt contains scikit-learn "
            "and redeploy the application."
        )

        st.stop()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        # First try normal pickle
        with model_path.open("rb") as file:
            model = pickle.load(file)

    except Exception as pickle_error:

        # If pickle fails, try joblib
        try:
            model = joblib.load(model_path)

        except Exception as joblib_error:

            st.error("❌ Unable to load the Logistic Regression model.")

            st.write("### Pickle loading error:")
            st.code(
                f"{type(pickle_error).__name__}: {pickle_error}"
            )

            st.write("### Joblib loading error:")
            st.code(
                f"{type(joblib_error).__name__}: {joblib_error}"
            )

            st.warning(
                "The model file may be corrupted or may not have been "
                "created using pickle.dump() or joblib.dump()."
            )

            st.info(
                "Regenerate logistic_regression_model.pkl from your "
                "training notebook and upload the new file to GitHub."
            )

            st.stop()

    # ========================================================
    # LOAD SCALER
    # ========================================================

    try:

        with scaler_path.open("rb") as file:
            scaler = pickle.load(file)

    except Exception as pickle_error:

        try:
            scaler = joblib.load(scaler_path)

        except Exception as joblib_error:

            st.error("❌ Unable to load the scaler.")

            st.write("### Pickle loading error:")
            st.code(
                f"{type(pickle_error).__name__}: {pickle_error}"
            )

            st.write("### Joblib loading error:")
            st.code(
                f"{type(joblib_error).__name__}: {joblib_error}"
            )

            st.warning(
                "The scaler file may be corrupted or saved using "
                "a different serialization method."
            )

            st.stop()

    # ========================================================
    # VALIDATE MODEL
    # ========================================================

    if not hasattr(model, "predict_proba"):

        st.error(
            "❌ The loaded model does not have predict_proba()."
        )

        st.write("Loaded object type:")
        st.code(str(type(model)))

        st.stop()

    # ========================================================
    # VALIDATE SCALER
    # ========================================================

    if not hasattr(scaler, "transform"):

        st.error(
            "❌ The loaded scaler does not have transform()."
        )

        st.write("Loaded object type:")
        st.code(str(type(scaler)))

        st.stop()

    # ========================================================
    # SHOW SUCCESS
    # ========================================================

    with st.expander("✅ Loaded Artifact Details"):

        st.write("**Model type:**")
        st.code(str(type(model)))

        st.write("**Scaler type:**")
        st.code(str(type(scaler)))

    return model, scaler


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model, scaler = load_artifacts()


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]


# ============================================================
# ENCODING
# ============================================================

GENDER = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

SMOKING = {
    "Never smoked": 4,
    "Current smoker": 1,
    "Former smoker": 3,
    "Ever smoked": 2,
    "Not current": 5,
    "No information": 0
}


# ============================================================
# DEFAULT VALUES
# ============================================================

DEFAULTS = dict(
    gender="Female",
    age=35,
    bmi=30.0,
    hba1c=7.0,
    glucose=180,
    smoking="Never smoked",
    htn=False,
    hd=False
)


# ============================================================
# CSS
# ============================================================

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=DM+Sans:wght@400;500;600&display=swap');

:root{
    --ink:#1D2646;
    --mute:#68729A;
    --line:#E0E5F5;
    --teal:#0FA39A;
    --violet:#6B5CE7;
    --warn:#E5484D
}

html,body,[class*="css"],.stApp{
    font-family:'DM Sans',sans-serif;
    color:var(--ink)
}

.stApp{
    background:
    radial-gradient(600px 400px at 90% -5%,#DDF4F1,transparent 70%),
    radial-gradient(500px 400px at -5% 30%,#E6E2FF,transparent 70%),
    #F5F7FC
}

#MainMenu,
footer,
header[data-testid="stHeader"]{
    visibility:hidden;
    height:0
}

.block-container{
    max-width:1150px;
    padding-top:2rem
}

h1,h2,h3{
    font-family:'Bricolage Grotesque',sans-serif!important;
    letter-spacing:-.02em;
    color:var(--ink)
}

.brand{
    display:inline-block;
    background:#D8F3F0;
    color:var(--teal);
    font-weight:600;
    font-size:14px;
    padding:6px 14px;
    border-radius:99px
}

.hero h1{
    font-size:clamp(40px,6vw,68px);
    font-weight:800;
    margin:18px 0 8px;
    line-height:1.05
}

.hero h4{
    color:var(--violet);
    font-weight:500;
    margin:0 0 12px
}

.hero p{
    color:var(--mute);
    max-width:52ch
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(
    > div > div[data-testid="stVerticalBlock"] .card-mark
){
    background:rgba(255,255,255,.75);
    backdrop-filter:blur(14px);
    border:1px solid #fff;
    border-radius:26px;
    box-shadow:0 10px 34px -12px rgba(50,60,130,.22);
    padding:14px
}

.sub{
    color:var(--mute);
    font-size:15px;
    margin:-6px 0 12px
}

div[data-baseweb="select"]>div{
    border-radius:12px;
    border:1.5px solid var(--line);
    background:#F5F7FC
}

div[role="radiogroup"]{
    gap:6px;
    background:#F5F7FC;
    padding:4px;
    border-radius:14px
}

div[role="radiogroup"] label{
    background:transparent;
    padding:4px 14px;
    border-radius:11px
}

div[data-testid="stSlider"] [role="slider"]{
    background:#fff;
    border:4px solid var(--violet)
}

.stButton>button{
    border-radius:99px;
    font-weight:700;
    padding:.75rem 1.4rem;
    transition:all .25s;
    border:1.5px solid var(--line);
    color:var(--mute);
    background:#fff
}

.stButton>button:hover{
    border-color:var(--violet);
    color:var(--violet);
    transform:translateY(-2px)
}

.stButton>button[kind="primary"]{
    color:#fff;
    border:0;
    font-size:17px;
    background:linear-gradient(
        100deg,
        var(--violet),
        var(--teal)
    );
    background-size:160% 100%;
    box-shadow:0 14px 26px -12px var(--violet)
}

.stButton>button[kind="primary"]:hover{
    background-position:100% 0;
    color:#fff;
    box-shadow:0 18px 30px -12px var(--violet)
}

.res{
    text-align:center
}

.ring{
    position:relative;
    width:210px;
    height:210px;
    margin:8px auto 14px
}

.ring svg{
    transform:rotate(-90deg);
    width:100%;
    height:100%
}

.ring circle{
    fill:none;
    stroke-width:16;
    stroke-linecap:round
}

.ring .bg{
    stroke:var(--line)
}

.ring .fg{
    stroke-dasharray:502;
    animation:fill 1.3s cubic-bezier(.2,.8,.2,1) both
}

@keyframes fill{
    from{
        stroke-dashoffset:502
    }
}

.ring .n{
    position:absolute;
    inset:0;
    display:grid;
    place-content:center;
    font:800 46px 'Bricolage Grotesque'
}

.ring .n small{
    font:500 13px 'DM Sans';
    color:var(--mute)
}

.badge{
    display:inline-block;
    padding:8px 18px;
    border-radius:99px;
    font-weight:700;
    background:#ECE9FF;
    color:var(--violet)
}

.low .badge{
    background:#D8F3F0;
    color:#087A73
}

.high .badge{
    background:#FDE7E8;
    color:#C0282D
}

.meter{
    height:10px;
    border-radius:9px;
    margin:20px 0 6px;
    position:relative;
    background:linear-gradient(
        90deg,
        #0FA39A,
        #F5C451,
        #E5484D
    )
}

.meter i{
    position:absolute;
    top:-5px;
    width:6px;
    height:20px;
    border-radius:4px;
    background:var(--ink);
    border:2px solid #fff
}

.expl{
    font-size:14.5px;
    color:var(--mute)
}

.note{
    font-size:12.5px;
    background:#FFF8E6;
    color:#7A5A00;
    border-radius:14px;
    padding:10px 14px;
    text-align:left;
    margin-top:14px
}

.step{
    background:#fff;
    border:1.5px solid var(--line);
    border-radius:22px;
    padding:18px;
    height:100%
}

.step b{
    display:inline-grid;
    place-items:center;
    width:34px;
    height:34px;
    border-radius:12px;
    background:#ECE9FF;
    color:var(--violet);
    margin-bottom:8px
}

.step p{
    color:var(--mute);
    font-size:14px;
    margin:4px 0 0
}

.disc{
    margin-top:26px;
    padding:20px 26px;
    border-radius:24px;
    background:linear-gradient(
        100deg,
        #ECE9FF,
        #D8F3F0
    )
}

details,
[data-testid="stExpander"]{
    border-radius:20px!important;
    background:#fff
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.session_state.setdefault("result", None)


# ============================================================
# RESET
# ============================================================

def reset():

    for k, v in DEFAULTS.items():
        st.session_state[k] = v

    st.session_state.result = None


# ============================================================
# PREDICTION
# ============================================================

def predict():

    s = st.session_state

    row = pd.DataFrame(
        [[
            GENDER[s.gender],
            s.age,
            int(s.htn),
            int(s.hd),
            SMOKING[s.smoking],
            s.bmi,
            s.hba1c,
            s.glucose
        ]],
        columns=FEATURES
    )

    # Scale exactly as done during training
    scaled = scaler.transform(row)

    probability = model.predict_proba(scaled)[0][1]

    return float(probability)


# ============================================================
# HERO
# ============================================================

h1, h2 = st.columns([1.2, 1])

with h1:

    st.markdown(
        """
        <div class="hero">
            <span class="brand">● GlucoLens</span>

            <h1>Diabetes Prediction</h1>

            <h4>AI-powered Diabetes Risk Assessment</h4>

            <p>
            This application uses a trained machine learning model
            to estimate diabetes risk from a handful of health
            parameters, including age, BMI, HbA1c and blood glucose.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with h2:

    st.markdown(
        """
        <svg viewBox="0 0 400 300"
             style="width:100%;max-width:380px;display:block;margin:auto">

        <defs>

        <linearGradient id="g"
                        x1="0"
                        y1="0"
                        x2="1"
                        y2="1">

            <stop offset="0"
                  stop-color="#6B5CE7"/>

            <stop offset="1"
                  stop-color="#0FA39A"/>

        </linearGradient>

        <clipPath id="c">

            <path d="
            M200 20
            C200 20 100 140 100 205
            a100 100 0 0 0 200 0
            C300 140 200 20 200 20Z"/>

        </clipPath>

        </defs>

        <path d="
        M200 20
        C200 20 100 140 100 205
        a100 100 0 0 0 200 0
        C300 140 200 20 200 20Z"
        fill="#fff"
        stroke="#DCE1F5"
        stroke-width="3"/>

        <g clip-path="url(#c)">

        <path d="
        M60 170
        q35-25 70 0
        t70 0
        70 0
        70 0
        V300H60Z"
        fill="url(#g)"/>

        </g>

        <path d="
        M140 220
        h36
        l14-30
        24 58
        18-38
        h34"
        fill="none"
        stroke="#fff"
        stroke-width="6"
        stroke-linecap="round"
        stroke-linejoin="round"/>

        </svg>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN
# ============================================================

left, right = st.columns([1.6, 1], gap="large")


with left:

    with st.container(border=True):

        st.markdown(
            '<span class="card-mark"></span>',
            unsafe_allow_html=True
        )

        st.markdown(
            "## Patient health information"
        )

        st.markdown(
            """
            <p class="sub">
            Adjust the sliders and options.
            Example values are pre-filled.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.radio(
            "Gender",
            list(GENDER),
            key="gender",
            horizontal=True,
            help="Biological sex as recorded in the training data."
        )

        c1, c2 = st.columns(2)

        c1.slider(
            "Age (years)",
            1,
            100,
            key="age",
            help="Risk rises with age, especially after 45."
        )

        c2.slider(
            "BMI (kg/m²)",
            10.0,
            70.0,
            step=0.1,
            key="bmi",
            help="18.5–24.9 is considered a healthy range."
        )

        c1.slider(
            "HbA1c level (%)",
            3.5,
            9.0,
            step=0.1,
            key="hba1c",
            help="Average blood sugar over 2–3 months."
        )

        c2.slider(
            "Blood glucose (mg/dL)",
            80,
            300,
            key="glucose",
            help="Current blood sugar reading."
        )

        st.selectbox(
            "Smoking history",
            list(SMOKING),
            key="smoking"
        )

        t1, t2 = st.columns(2)

        t1.toggle(
            "Hypertension",
            key="htn",
            help="Diagnosed high blood pressure."
        )

        t2.toggle(
            "Heart disease",
            key="hd",
            help="Any diagnosed heart condition."
        )

        b1, b2 = st.columns([2.2, 1])

        go = b1.button(
            "Predict Diabetes Risk",
            type="primary",
            use_container_width=True
        )

        b2.button(
            "Reset",
            on_click=reset,
            use_container_width=True
        )


    if go:

        with st.spinner(
            "Analysing your health data…"
        ):

            time.sleep(0.9)

            try:

                st.session_state.result = predict()

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    f"{type(e).__name__}: {e}"
                )


# ============================================================
# RESULT
# ============================================================

with right:

    p = st.session_state.result

    with st.container(border=True):

        st.markdown(
            '<span class="card-mark"></span>',
            unsafe_allow_html=True
        )

        if p is None:

            cls = ""
            pct = "--"
            badge = "Awaiting input"
            offset = 502
            pos = 0

            text = (
                "Fill in the health details and press "
                "Predict Diabetes Risk to see your estimate."
            )

            color = "#68729A"

        else:

            high = p >= 0.5

            cls = "high" if high else "low"

            pct = f"{round(p * 100)}%"

            badge = (
                "🔴 Higher Risk"
                if high
                else
                "🟢 Low Risk"
            )

            offset = 502 * (1 - p)

            pos = round(p * 100)

            color = (
                "#E5484D"
                if high
                else
                "#0FA39A"
            )

            if high:

                text = (
                    "The model predicts a higher likelihood of "
                    "diabetes for these values. Consider discussing "
                    "these measurements with a healthcare professional."
                )

            else:

                text = (
                    "The model predicts a low likelihood of diabetes "
                    "for these values. Healthy habits and regular "
                    "check-ups can support overall health."
                )

        st.markdown(
            f"""
            <div class="res {cls}">

            <h3>Prediction Result</h3>

            <div class="ring">

                <svg viewBox="0 0 180 180">

                    <circle
                        class="bg"
                        cx="90"
                        cy="90"
                        r="80"/>

                    <circle
                        class="fg"
                        cx="90"
                        cy="90"
                        r="80"
                        stroke="{color}"
                        style="stroke-dashoffset:{offset}"/>

                </svg>

                <div class="n">

                    <span>{pct}</span>

                    <small>model probability</small>

                </div>

            </div>

            <span class="badge">{badge}</span>

            <div class="meter">

                <i style="left:calc({pos}% - 3px)"></i>

            </div>

            <p class="expl">
                {text}
            </p>

            <div class="note">

                <b>Not a diagnosis.</b>

                This is a machine-learning prediction,
                not a medical diagnosis.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("### How it works")

steps = [
    ("1", "Enter", "Add your health information."),
    ("2", "Process", "The machine learning model processes the inputs."),
    ("3", "Predict", "The model generates a prediction."),
    ("4", "Review", "The result is displayed with a risk indicator.")
]


for col, (n, title, description) in zip(
    st.columns(4),
    steps
):

    col.markdown(
        f"""
        <div class="step">

            <b>{n}</b>

            <h3>{title}</h3>

            <p>{description}</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.write("")

with st.expander("About the model"):

    st.write(
        """
        Predictions come from a trained Logistic Regression
        classification model, trained on a public dataset of
        100,000 patient records with eight features.

        Inputs are standardised with StandardScaler, and class
        weights are balanced so the model can better identify
        positive diabetes cases.
        """
    )

    m1, m2, m3 = st.columns(3)

    m1.metric("Accuracy", "88.6%")
    m2.metric("Recall", "88.9%")
    m3.metric("Precision", "43.3%")


# ============================================================
# MEASUREMENT INFORMATION
# ============================================================

with st.expander("What do the measurements mean?"):

    st.write(
        """
        HbA1c reflects average blood sugar over the past
        two to three months.

        Blood glucose represents the blood sugar level
        measured at a particular time.

        BMI compares weight to height.
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disc">

    <b>Important disclaimer.</b>

    This tool is intended for educational and informational
    purposes only. It should not be used as a substitute for
    professional medical advice, diagnosis, or treatment.

    </div>
    """,
    unsafe_allow_html=True
)
