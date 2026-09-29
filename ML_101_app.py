import joblib
import pandas as pd
import streamlit as st

# -------------------------------
# Streamlit
# -------------------------------
#"""Lädt das serialisierte Modell und stellt ein Web-Interface für Einzelvorhersagen bereit."""
model = joblib.load('best_personality_model.joblib')
st.latex("E = mc^3^E=")
st.divider()
st.title('Personality Profile Predictor') 

col1, col2 = st.columns(2)
with col1:
  age = st.slider('Age', 10, 100, 25)
  gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
with col2:
  hand = st.selectbox('Hand', ['Right', 'Left', 'Both'])

input_dict = {'age': age, 'gender': gender, 'hand': hand}

# st.subheader('Likert-Skalen (1-5)')
# num_cols = model.named_steps['preprocessor'].transformers_[0][2]

# for col in sorted(
#     (c for c in num_cols if c != 'age'), key=lambda x: (x[0], int(x[1:]))
# ):
#   input_dict[col] = st.slider(col, 1, 5, 3)

QUESTIONS_ORDER = {
    'N1': 'I get stressed out easily. (Ich gerate leicht unter Stress.)',
    'N2': 'I am relaxed most of the time. (Ich bin die meiste Zeit entspannt.)',
    'N3': 'I worry about things. (Ich mache mir oft Sorgen.)',
    'N4': 'I seldom feel blue. (Ich bin selten niedergeschlagen.)',
    'N5': 'I am easily disturbed. (Ich lasse mich leicht aus der Ruhe bringen.)',
    'N6': 'I get upset easily. (Ich rege mich schnell auf.)',
    'N7': 'I change my mood a lot. (Meine Laune ändert sich oft.)',
    'N8': 'I have frequent mood swings. (Ich habe häufige Stimmungsschwankungen.)',
    'N9': 'I get irritated easily. (Ich bin schnell gereizt.)',
    'N10': 'I often feel blue. (Ich fühle mich oft niedergeschlagen.)',
    'E1': 'I am the life of the party. (Ich stehe gerne im Mittelpunkt jeder Party.)',
    'E3': 'I feel comfortable around people. (Ich fühle mich wohl unter Menschen.)',
    'E4': 'I keep in the background. (Ich halte mich im Hintergrund.)',
    'E5': 'I start conversations. (Ich beginne oft Gespräche.)',
    'E7': 'I talk to a lot of different people at parties. (Auf Feiern spreche ich mit vielen verschiedenen Leuten.)',
    'E9': "I don't mind being the center of attention. (Es stört mich nicht, im Mittelpunkt der Aufmerksamkeit zu stehen.)",
    'E10': 'I am quiet around strangers. (In der Nähe von Fremden bin ich eher ruhig.)',
    'C4': 'C4: I make a mess of things. (Ich bringe Dinge oft durcheinander.)',
    'A4': "I sympathize with others' feelings. (Ich habe Mitgefühl für die Gefühle anderer.)",
}

st.subheader('Likert-Skalen (1-5)')
num_cols = model.named_steps['preprocessor'].transformers_[0][2]

for col, text in QUESTIONS_ORDER.items():
  if col in num_cols:
    input_dict[col] = st.slider(f'{col}: {text}', 1, 5, 3)

if st.button('Predict Profile'):
  pred = model.predict(pd.DataFrame([input_dict]))[0]
  st.success(f'Vorhergesagtes Persönlichkeitsprofil: **{pred}**')



  