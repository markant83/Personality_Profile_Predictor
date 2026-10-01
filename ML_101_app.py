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

# -------------------------------
# Hilfsfunktionen & Mappings
# -------------------------------
def bilingual_label(en: str, de: str):
    st.markdown(
        f'<span style="font-weight: 600; font-size: 1rem;">{en}</span> '
        f'<span style="font-size: 0.82rem; color: #888; margin-left: 6px;">({de})</span>',
        unsafe_allow_html=True,
    )

gender_labels = {
    "Female": "Female (Weiblich)",
    "Male": "Male (Männlich)",
    "Other": "Other (Divers / Sonstige)",
}

hand_labels = {
    "Right": "Right (Rechtshänder)",
    "Left": "Left (Linkshänder)",
    "Both": "Both (Beidhänder)",
}

# -------------------------------
# Benutzeroberfläche
# -------------------------------
st.title("🧠 Personality Profile Predictor")
st.caption("Big Five / IPIP psychometrische Persönlichkeitsklassifikation")
st.divider()

col1, col2 = st.columns(2)
with col1:
    bilingual_label("Gender", "Geschlecht")
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male", "Other"],
        format_func=lambda x: gender_labels.get(x, x),
        label_visibility="collapsed",
    )

    bilingual_label("Age", "Alter")
    age = st.slider("Age", 10, 100, 45, label_visibility="collapsed")

with col2:
    bilingual_label("Handedness", "Händigkeit")
    hand = st.selectbox(
        "Hand",
        options=["Right", "Left", "Both"],
        format_func=lambda x: hand_labels.get(x, x),
        label_visibility="collapsed",
    )

input_dict = {"age": age, "gender": gender, "hand": hand}

st.subheader("Likert-Skalen (1–5)")
st.markdown(
    '<p style="font-style: italic; margin-top: -10px; margin-bottom: 20px; color: #888;">'
    "1 = Disagree (Trifft nicht zu) &nbsp;|&nbsp; "
    "2 = Slightly Disagree &nbsp;|&nbsp; "
    "3 = Neutral &nbsp;|&nbsp; "
    "4 = Slightly Agree &nbsp;|&nbsp; "
    "5 = Agree (Trifft voll zu)"
    "</p>",
    unsafe_allow_html=True,
)

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

num_cols = model.named_steps["preprocessor"].transformers_[0][2]

for col, (en_text, de_text) in QUESTIONS.items():
    if col in num_cols:
        st.markdown(
            f'<div style="margin-top: 10px; margin-bottom: -10px;">'
            f"<strong>{col}:</strong> {en_text} "
            f'<span style="font-size: 0.85em; color: #888;">({de_text})</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        input_dict[col] = st.slider(col, 1, 5, 3, label_visibility="collapsed")

st.divider()

explanations = {
    "Resilient": "Emotional stabil, anpassungsfähig und stressresistent.",
    "Undercontroller": "Eher impulsiv, spontan und geringe Selbstkontrolle.",
    "Overcontroller": "Sehr gewissenhaft, gehemmt und sicherheitsorientiert.",
    "Moderate": "Ausgeglichenes Durchschnittsprofil ohne extreme Ausschläge.",
}

# -------------------------------
# Inferenz & Ausgabe
# -------------------------------
if st.button("Predict Profile (Profil berechnen)", type="primary"):
    pred = model.predict(pd.DataFrame([input_dict]))[0]
    st.success(f"Vorhergesagtes Persönlichkeitsprofil: **{pred}**")

    beschreibung = explanations.get(str(pred), "Keine Beschreibung verfügbar.")
    st.info(f"ℹ️ **Bedeutung:** {beschreibung}")

  