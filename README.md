# 🧠 Personality Profile Predictor (Big Five / IPIP)

Eine interaktive Machine-Learning-Webanwendung auf Basis von **Streamlit** und **Scikit-Learn**, die anhand von Persönlichkeitsfragen (IPIP / Big Five Inventory) psychometrische Persönlichkeitsprofile klassifiziert.

---

## 📌 Übersicht & Funktionsweise

### 1. Das Machine Learning Modell
* **Datengrundlage:** ~20.000 Umfragedatensätze basierend auf Likert-Skalen (1–5) sowie demografischen Merkmalen (Alter, Geschlecht).
* **Datenbereinigung:**
  * Automatische Erkennung und Umrechnung von Geburtsjahren in das tatsächliche Alter.
  * Filterung unplausibler Extremwerte (Alter > 120).
  * Validierung der Likert-Werte (Bereich 1 bis 5) mit Median-Imputation.
  * Multikollinearitätsprüfung mittels Pearson-Korrelationsmatrix (|r| > 0.85).
* **Pipeline-Architektur:**
  * Numerische Merkmale: Median-Imputation (`SimpleImputer`) + Standardisierung (`StandardScaler`).
  * Kategoriale Merkmale: Modus-Imputation (`most_frequent`) + One-Hot-Encoding (`OneHotEncoder`).
* **Modellvergleich & Tuning:**
  * Vergleich via 5-Fold Cross-Validation: Logistic Regression, Random Forest und HistGradientBoosting.
  * Sieger-Modell: **HistGradientBoostingClassifier**.
  * Hyperparameter-Optimierung über Rastersuche (`GridSearchCV`) optimiert auf **F1-Macro-Score** (ungewichtetes Mittel über alle 5 Persönlichkeitsklassen):
    * **Bester F1-Macro CV-Score:** `~0.7944`
    * **Test-Set Accuracy (Gesamttrefferquote):** `~86.1 %`
---

### 2. Die Anwendung & der Persönlichkeitstest
Die Streamlit-App (`ML_101_app.py`) dient als Frontend für das exportierte Modell:
1. **Interaktive Eingabe:** Nutzer beantworten Fragen auf einer 5-stufigen Likert-Skala (von *Trifft gar nicht zu* bis *Trifft voll zu*) und geben demografische Daten an.
2. **Echtzeit-Inferenz:** Die Eingaben werden strukturiert, durch die vortrainierte Pipeline transformiert und direkt klassifiziert.
3. **Ergebnisdarstellung:** Ausgabe des prognostizierten Persönlichkeitstyps inklusive Profilbeschreibung.

---

## 🚀 Lokale Installation & Ausführung

Folge diesen Schritten, um das Projekt lokal auszuführen:

### 1. Repository klonen
```bash
git clone [https://github.com/markant83/Personality_Profile_Predictor.git](https://github.com/markant83/Personality_Profile_Predictor.git)
cd Personality_Profile_Predictor
```

### 2. Virtuelle Umgebung erstellen & aktivieren
* **Windows:**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 4. Anwendung starten
```bash
streamlit run ML_101_app.py
```
Die App öffnet sich anschließend automatisch im Browser unter `http://localhost:8501`.

*(Optional)* Falls das Modell neu trainiert werden soll:
```bash
python ML_101_qwert.py
```

---

## 🛠 Tech-Stack
* **Sprache:** Python 3.10+
* **ML & Datenverarbeitung:** Scikit-Learn, Pandas, NumPy
* **Deployment & UI:** Streamlit, Joblib
* **Experiment-Tracking:** MLflow
