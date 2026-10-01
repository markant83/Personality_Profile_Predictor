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
    /* Styling für das kleinere, graue Deutsch in Selectboxen */
    div[data-baseweb="select"] span.de-sub {
        font-size: 0.82em;
        color: #888;
        margin-left: 5px;
    }
    /* Styling für den Predict-Button */
    div.stButton > button span.de-btn {
        font-size: 0.82em;
        opacity: 0.8;
        font-weight: normal;
        margin-left: 6px;
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
        "Resilient",
        "Widerstandsfähig / Stressresistent",
        "Emotionally stable, adaptable, and handles stress well.",
        "Emotional stabil, anpassungsfähig und stressresistent.",
    ),
    "Undercontroller": (
        "Undercontroller",
        "Impulsiv / Gering reguliert",
        "More impulsive, spontaneous, and prone to low self-regulation.",
        "Eher impulsiv, spontan und geringe Selbstkontrolle.",
    ),
    "Overcontroller": (
        "Overcontroller",
        "Überkontrolliert / Gehemmt",
        "Highly conscientious, cautious, and security-oriented.",
        "Sehr gewissenhaft, gehemmt und sicherheitsorientiert.",
    ),
    "Moderate": (
        "Moderate",
        "Ausgeglichen / Durchschnittlich",
        "Balanced average profile without prominent extremes.",
        "Ausgeglichenes Durchschnittsprofil ohne extreme Ausschläge.",
    ),
}

# -------------------------------
# Benutzeroberfläche
# -------------------------------
st.markdown(
    '<h2>🧠 Personality Profile Predictor '
    '<span style="font-size: 0.6em; color: #888; font-weight: normal;">(Persönlichkeitsprofil-Vorhersage)</span></h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color: #666; font-size: 0.9em; margin-top: -10px;">'
    'Big Five / IPIP Psychometric Personality Classification '
    '<span style="color: #888;">(Psychometrische Persönlichkeitsklassifikation)</span></p>',
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
button_clicked = st.button("Predict Profile (Profil berechnen)", type="primary")

if button_clicked:
    pred = str(model.predict(pd.DataFrame([input_dict]))[0])
    en_title, de_title, en_desc, de_desc = PROFILES.get(
        pred, (pred, pred, "No description available.", "Keine Beschreibung verfügbar.")
    )

    # Zweisprachige Ergebnis-Box
    st.markdown(
        f"""
        <div style="background-color: rgba(46, 125, 50, 0.12); border-left: 5px solid #2e7d32; padding: 14px; border-radius: 4px; margin-bottom: 15px;">
            <div style="font-size: 0.9rem; color: #555; margin-bottom: 4px;">
                <strong>Predicted Personality Profile</strong> <span style="font-size: 0.82em; color: #888;">(Vorhergesagtes Persönlichkeitsprofil)</span>
            </div>
            <div style="font-size: 1.45rem; font-weight: bold; color: #1b5e20;">
                {en_title} <span style="font-size: 0.72em; font-weight: normal; color: #666;">({de_title})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Zweisprachige Erklärung
    st.markdown(
        f"""
        <div style="background-color: rgba(33, 150, 243, 0.08); border-left: 5px solid #1976d2; padding: 12px 14px; border-radius: 4px;">
            <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">
                ℹ️ Meaning <span style="font-size: 0.82em; color: #888; font-weight: normal;">(Bedeutung)</span>:
            </div>
            <div style="font-size: 0.95rem; line-height: 1.5;">
                {en_desc} <span style="font-size: 0.85em; color: #777;">({de_desc})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  