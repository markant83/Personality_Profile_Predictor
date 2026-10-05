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
Die Streamlit-App (`app.py`) dient als Frontend für das exportierte Modell:
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
streamlit run app.py
```
Die App öffnet sich anschließend automatisch im Browser unter `http://localhost:8501`.

*(Optional)* Falls das Modell neu trainiert werden soll:
```bash
python model.py
```

---

## 🛠 Tech-Stack
* **Sprache:** Python 3.10+
* **ML & Datenverarbeitung:** Scikit-Learn, Pandas, NumPy
* **Deployment & UI:** Streamlit, Joblib
* **Experiment-Tracking:** MLflow


# 🧠 Personality Profile Predictor (Big Five / IPIP)

Eine interaktive Machine-Learning-Webanwendung auf Basis von **Streamlit** und **Scikit-Learn**, die anhand eines kurzen psychometrischen Fragebogens Persönlichkeitsprofile prognostiziert.

---

## 📌 1. Überblick (Overview)

### Was das Projekt tut
Die Anwendung prognostiziert den psychometrischen Persönlichkeitstyp einer Person anhand eines kurzen Fragebogens sowie grundlegender demografischer Angaben.

### Das Problem
Es handelt sich um eine überwachte Mehrklassen-Klassifikation (*Supervised Multiclass Classification*). Ziel ist es, eingehende Selbsteinschätzungen anhand psychologischer Konstrukte automatisch und stabil einer übergeordneten Persönlichkeitskategorie zuzuordnen.

### Der Datensatz
* **Herkunft & Repräsentation:** Umfragedatensatz basierend auf dem International Personality Item Pool (IPIP / Big Five Inventory) mit ca. 20.000 Beobachtungen. Jede Zeile repräsentiert eine befragte Person.
* **Merkmale (Features):** 19 Persönlichkeitsfragen auf einer 5-stufigen Likert-Skala (1 = *Trifft gar nicht zu* bis 5 = *Trifft voll zu*) sowie demografische Merkmale (Alter, Geschlecht).
* **Zielklassen (Targets):** Vier psychometrische Prototypen:
  * 🟢 **Moderat** (*Average*)
  * 🔵 **Resilient**
  * 🟡 **Überkontrolliert** (*Overcontrolled*)
  * 🔴 **Unterkontrolliert** (*Undercontrolled*)

### Der Ansatz
Strukturierter End-to-End-Workflow:
1. **EDA:** Plausibilisierung von Geburtsjahren, Bereinigung unplausibler Extremwerte (Alter > 120) und Multikollinearitätsprüfung mittels Pearson-Korrelationsmatrix (|r| < 0.85).
2. **Vorverarbeitungspipeline:**
   * Numerische Merkmale: Median-Imputation (`SimpleImputer`) + Standardisierung (`StandardScaler`).
   * Kategoriale Merkmale: Modus-Imputation (`SimpleImputer`, `most_frequent`) + One-Hot-Encoding (`OneHotEncoder`).
3. **Modellvergleich & Tuning:**
   * Vergleich via 5-Fold Cross-Validation: Logistic Regression, Random Forest und HistGradientBoosting.
   * Sieger-Modell: **HistGradientBoostingClassifier**.
   * Hyperparameter-Optimierung über Rastersuche (`GridSearchCV`) optimiert auf **F1-Macro-Score** (ungewichtetes Mittel über alle 5 Persönlichkeitsklassen):
4. **Speichern des besten Modells:** Export der trainierten Gesamt-Pipeline als serialisierte Datei (`joblib`).
5. **Streamlit-App:** Bereitstellung des Modells in einem interaktiven Webinterface (`app.py`).

### Das Ergebnis
* **Bestes Modell:** `HistGradientBoostingClassifier`
* **Optimierungsmetrik:** F1-Macro-Score (ungewichteter Mittelwert über alle Zielklassen)
* **Performance:**
  * Bester Cross-Validation-Score (F1-Macro): **~0.7944**
  * Test-Set Accuracy: **~86.1 %** auf ungesehenen Testdaten

### So verwendest du die App
Der Benutzer beantwortet die 19 Fragen im Webinterface auf einer 5-stufigen Skala und gibt Alter sowie Geschlecht an. Nach der Eingabe führt die App eine Sofort-Inferenz durch und liefert den vorhergesagten Persönlichkeitstyp inklusive detaillierter Profilbeschreibung.

---

## 🚀 2. Einrichtung (Setup)

Schritt-für-Schritt-Anleitung zur vollständigen Reproduktion des Projekts:

### 1. Repository klonen
```bash
git clone [https://github.com/markant83/Personality_Profile_Predictor.git](https://github.com/markant83/Personality_Profile_Predictor.git)
cd Personality_Profile_Predictor
```

### 2. Daten beschaffen
1. Lade den Rohdatensatz über folgenden Google-Drive-Link herunter:
   👉 `https://drive.google.com/file/d/1Qnc2hC9szsKlHiutJIttwhnxphGffn-2/view?usp=drive_link`
2. Erstelle im Projektordner das Verzeichnis `data/` (sofern noch nicht vorhanden):
   ```bash
   mkdir data
   ```
3. Lege die heruntergeladene CSV-Datei direkt in diesem Ordner ab (z. B. als `data/data-final.csv`).

### 3. Umgebung erstellen & aktivieren
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

### 4. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 5. Notebooks / Trainingspipeline ausführen
Führe das Trainingsskript bzw. die Notebooks in dieser Reihenfolge aus, um die Datenexploration, Vorverarbeitung, das Modelltraining und die Modellerstellung unter Nutzung fester Zufalls-States (`random_state=42`) vollständig zu reproduzieren:
```bash
python PPP_model.py
```
*(Das trainierte Modell wird dabei als serialisierte Datei im Repository-Verzeichnis gespeichert.)*

### 6. Streamlit-App lokal starten
Starte das Webinterface mit folgendem Befehl:
```bash
streamlit run PPP_app.py
```
Die App öffnet sich anschließend automatisch im Browser unter `http://localhost:8501`.

---

## 🛠 Tech-Stack

* **Sprache:** Python 3.10+
* **ML & Datenverarbeitung:** Scikit-Learn, Pandas, NumPy
* **UI & Deployment:** Streamlit, Joblib
* **Experiment-Tracking:** MLflow
