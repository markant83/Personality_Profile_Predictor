import joblib
import pandas as pd
import streamlit as st

# -------------------------------
# Konfiguration & Modell laden
# -------------------------------
st.set_page_config(
    page_title="Personality Profile Predictor",
    page_icon="🧠",
    layout="centered",
)

model = joblib.load("best_personality_model.joblib")

# Globale CSS-Stile: Dropdowns & Button zweisprachig stylen
st.markdown(
    """
    <style>
    /* 1. Oberen Leerraum reduzieren */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 920px !important; /* Optimale Breite für einzeilige Skalen */
    }

    /* 2. Plastischer 3D-Button (etwas größer & dezent grau) */
    div.stButton > button {
        background: linear-gradient(180deg, #3a3f47 0%, #2b2f36 100%) !important;
        color: #ffffff !important;
        border: 1px solid #1f2227 !important;
        border-bottom: 4px solid #16181b !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.1s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(180deg, #444a54 0%, #323740 100%) !important;
        border-color: #2b2f36 !important;
    }
    div.stButton > button:active {
        border-bottom-width: 1px !important;
        transform: translateY(3px) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
    }

    /* 3. Dropdown-Schriftgröße vergrößern */
    div[data-baseweb="select"] {
        font-size: 1.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Hilfsfunktionen & Wörterbücher
# -------------------------------
def bilingual_label(en: str, de: str):
    st.markdown(
        f'<span style="font-weight: 600; font-size: 1rem;">{en}</span> '
        f'<span style="font-size: 0.82rem; color: #888; margin-left: 6px;">({de})</span>',
        unsafe_allow_html=True,
    )

GENDER_OPTIONS = ["Female", "Male", "Other"]
GENDER_DE = {"Female": "Weiblich", "Male": "Männlich", "Other": "Divers / Sonstige"}

HAND_OPTIONS = ["Right", "Left", "Both"]
HAND_DE = {"Right": "Rechtshänder", "Left": "Linkshänder", "Both": "Beidhänder"}

QUESTIONS = {
    "N1": ("I get stressed out easily.", "Ich gerate leicht unter Stress."),
    "N2": ("I am relaxed most of the time.", "Ich bin die meiste Zeit entspannt."),
    "N3": ("I worry about things.", "Ich mache mir oft Sorgen."),
    "N4": ("I seldom feel blue.", "Ich bin selten niedergeschlagen."),
    "N5": ("I am easily disturbed.", "Ich lasse mich leicht aus der Ruhe bringen."),
    "N6": ("I get upset easily.", "Ich rege mich schnell auf."),
    "N7": ("I change my mood a lot.", "Meine Laune ändert sich oft."),
    "N8": ("I have frequent mood swings.", "Ich habe häufige Stimmungsschwankungen."),
    "N9": ("I get irritated easily.", "Ich bin schnell gereizt."),
    "N10": ("I often feel blue.", "Ich fühle mich oft niedergeschlagen."),
    "E1": ("I am the life of the party.", "Ich stehe gerne im Mittelpunkt jeder Party."),
    "E3": ("I feel comfortable around people.", "Ich fühle mich wohl unter Menschen."),
    "E4": ("I keep in the background.", "Ich halte mich im Hintergrund."),
    "E5": ("I start conversations.", "Ich beginne oft Gespräche."),
    "E7": ("I talk to a lot of different people at parties.", "Auf Feiern spreche ich mit vielen verschiedenen Leuten."),
    "E9": ("I don't mind being the center of attention.", "Es stört mich nicht, im Mittelpunkt der Aufmerksamkeit zu stehen."),
    "E10": ("I am quiet around strangers.", "In der Nähe von Fremden bin ich eher ruhig."),
    "C4": ("I make a mess of things.", "Ich bringe Dinge oft durcheinander."),
    "A4": ("I sympathize with others' feelings.", "Ich habe Mitgefühl für die Gefühle anderer."),
}

PROFILES = {
    "Resilient": (
        "Emotionally stable, adaptable, and handles stress well.",
        "Emotional stabil, anpassungsfähig und stressresistent.",
    ),
    "Undercontroller": (
        "More impulsive, spontaneous, and prone to low self-regulation.",
        "Eher impulsiv, spontan und geringe Selbstkontrolle.",
    ),
    "Overcontroller": (
        "Highly conscientious, cautious, and security-oriented.",
        "Sehr gewissenhaft, gehemmt und sicherheitsorientiert.",
    ),
    "Moderate": (
        "Balanced average profile without prominent extremes.",
        "Ausgeglichenes Durchschnittsprofil ohne extreme Ausschläge.",
    ),
}

# -------------------------------
# Benutzeroberfläche
# -------------------------------
st.markdown(
    '<h2>🧠 Personality Profile Predictor ',
    unsafe_allow_html=True,
)
st.markdown(
    '<span style="font-size: 1.6em; color: #888; font-weight: normal;">(Persönlichkeitsprofil-Vorhersage)</span></h2>',
    unsafe_allow_html=True,
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    bilingual_label("Gender", "Geschlecht")
    gender = st.selectbox(
        "Gender",
        options=GENDER_OPTIONS,
        format_func=lambda x: f"{x} ({GENDER_DE.get(x, '')})",
        label_visibility="collapsed",
    )

    bilingual_label("Age", "Alter")
    age = st.slider("Age", 10, 100, 45, label_visibility="collapsed")

with col2:
    bilingual_label("Handedness", "Händigkeit")
    hand = st.selectbox(
        "Hand",
        options=HAND_OPTIONS,
        format_func=lambda x: f"{x} ({HAND_DE.get(x, '')})",
        label_visibility="collapsed",
    )

input_dict = {"age": age, "gender": gender, "hand": hand}

bilingual_label("Likert Scales (1–5)", "Likert-Skalen 1–5")
st.markdown(
    '<p style="font-style: italic; margin-top: -5px; margin-bottom: 20px; color: #888; font-size: 0.88rem;">'
    "1 = Disagree <span style='font-size: 0.85em;'>(Trifft nicht zu)</span> &nbsp;|&nbsp; "
    "2 = Slightly Disagree <span style='font-size: 0.85em;'>(Trifft eher nicht zu)</span> &nbsp;|&nbsp; "
    "3 = Neutral <span style='font-size: 0.85em;'>(Neutral)</span> &nbsp;|&nbsp; "
    "4 = Slightly Agree <span style='font-size: 0.85em;'>(Trifft eher zu)</span> &nbsp;|&nbsp; "
    "5 = Agree <span style='font-size: 0.85em;'>(Trifft voll zu)</span>"
    "</p>",
    unsafe_allow_html=True,
)

num_cols = model.named_steps["preprocessor"].transformers_[0][2]

for col, (en_text, de_text) in QUESTIONS.items():
    if col in num_cols:
        st.markdown(
            f'<div style="margin-top: 12px; margin-bottom: -10px;">'
            f"<strong>{col}:</strong> {en_text} "
            f'<span style="font-size: 0.82em; color: #888;">({de_text})</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        input_dict[col] = st.slider(col, 1, 5, 3, label_visibility="collapsed")

st.divider()

# -------------------------------
# Inferenz & Ausgabe
# -------------------------------
# Button mit kleinerem deutschen Text
button_clicked = st.button("Predict Profile (Profil berechnen)")

if button_clicked:
    pred = str(model.predict(pd.DataFrame([input_dict]))[0])
    en_desc, de_desc = PROFILES.get(
        pred, ("No description available.", "Keine Beschreibung verfügbar.")
    )

    st.markdown(
        f"""
        <div style="display: flex; align-items: stretch; background-color: rgba(46, 125, 50, 0.25); 
                    border: 1px solid rgba(76, 175, 80, 0.5); border-left: 5px solid #4caf50; 
                    border-radius: 8px; margin-top: 15px; overflow: hidden;">
            <!-- Linke Spalte: Profilname -->
            <div style="flex: 0 0 30%; display: flex; align-items: center; justify-content: center; 
                        padding: 16px 20px; font-size: 1.5rem; font-weight: 700; color: #ffffff; text-align: center;">
                {pred}
            </div>
            <!-- Trennbalken -->
            <div style="width: 2px; background-color: #111111; opacity: 0.8;"></div>
            <!-- Rechte Spalte: Erklärung untereinander -->
            <div style="flex: 1; display: flex; flex-direction: column; justify-content: center; 
                        padding: 14px 22px; gap: 4px;">
                <div style="font-size: 1rem; color: #ffffff; line-height: 1.4;">
                    {en_desc}
                </div>
                <div style="font-size: 0.85rem; color: #cfcfcf; line-height: 1.4;">
                    ({de_desc})
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


