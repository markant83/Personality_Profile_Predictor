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
Um eine robuste Datenbasis für das Modelltraining sicherzustellen, durchlaufen die Rohdaten (~19.700 Einträge) eine mehrstufige Bereinigung und Validierung:

1. **Explorative Datenanalyse (EDA) & Datenbereinigung:**
   * **Altersbereinigung & Anomalie-Korrektur (`age`):**
     * *Geburtsjahr-Erkennung:* 73 Einträge im Wertebereich von 1920 bis 2026 wurden als Geburtsjahre identifiziert und automatisiert in das biologische Alter umgerechnet (`2026 - Geburtsjahr`).
     * *Extremwert-Filterung:* Unplausible Altersangaben (> 120 Jahre, z. B. Eingabefehler wie 999.999.999 oder 208) wurden identifiziert (9 Fälle) und auf `NaN` gesetzt, um spätere Modellverzerrungen zu vermeiden.
   * **Validierung der Likert-Skalen:**
     * Alle psychometrischen Fragebogen-Items (`N1`–`N10`, `E1`–`E10`, `C4`, `A4`) wurden auf den zulässigen Wertebereich von 1 bis 5 überprüft.
     * Eine Zeile mit unzulässigen Nullen (`0`) über alle Skalen hinweg wurde erkannt und auf `NaN` gesetzt. Fehlwerte in den numerischen Features werden in der Trainingspipeline robust via **Median-Imputation** ersetzt.
   * **Kategoriale Datenqualität & Imputationsstrategie:**
     * Analyse der kategorialen Merkmale ergab minimale Fehlwerte in `gender` (24 Fälle / 0,12 %) und `hand` (100 Fälle / 0,51 %).
     * Aufgrund der geringen Fehlquote (< 0,6 %) werden diese in der Pipeline deterministisch über den **Modus** (`most_frequent`: Female bzw. Right) imputiert und anschließend per One-Hot-Encoding transformiert.
   * **Multikollinearitätsprüfung:**
     * Berechnung der paarweisen Pearson-Korrelationsmatrix aller numerischen Prädiktoren mit einem Schwellenwert von |r| > 0.85.
     * Ergebnis: Keine redundanten Variablen vorhanden (höchste beobachtete Korrelation liegt deutlich darunter), womit alle Items im Feature-Set verbleiben.
   * **Zielklassen-Analyse (`target`):**
     * Identifikation einer Klassenimbalance im Zielmerkmal: *Moderate* (42,86 %), *Resilient* (31,25 %), *Overcontroller* (14,35 %) und *Undercontroller* (11,54 %).
     * Zur Wahrung der Klassenproportionen wird der Datensplit stratifiziert durchgeführt (`stratify=y`) und die Modellbewertung primär anhand des makro-gemittelten F1-Scores (`f1_macro`) vorgenommen.
2. **Vorverarbeitungspipeline:**
   * Numerische Merkmale: Median-Imputation (`SimpleImputer`) + Standardisierung (`StandardScaler`).
   * Kategoriale Merkmale: Modus-Imputation (`SimpleImputer`, `most_frequent`) + One-Hot-Encoding (`OneHotEncoder`).
3. **Modellvergleich & Tuning:**
   * Vergleich via 5-Fold Cross-Validation:
     * Logistic Regression
     * Random Forest
     * HistGradientBoosting
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

### Fazit
Nach der Evaluierung von drei Basis-Modellen (Logistic Regression mit Macro-F1 von 0,7671, Random Forest mit 0,7397 und HistGradientBoosting mit 0,7954) zeigte der HistGradientBoostingClassifier die stärkste anfängliche Leistung. Nach der Hyperparameter-Optimierung über GridSearchCV (beim systematischen Testen von 27 Kombinationen aus learning_rate, max_leaf_nodes und min_samples_leaf über eine 3-Fold Cross-Validation) erzielte die Konfiguration mit learning_rate=0.1, max_leaf_nodes=31 und min_samples_leaf=10 den besten mittleren CV-Score von 0,7944. Auf dem ungesehenen Testdatensatz erreichte die finale Pipeline eine Test-Accuracy von 86,13 %. Aufgrund der überlegenen Balance aus Klassen-F1-Werten, schneller Inferenzzeit und robuster Generalisierung wurde diese Pipeline als finales Modell für das Deployment via Streamlit ausgewählt.

### So verwendest du die App
Die Streamlit-App (`app.py`) dient als Frontend für das exportierte Modell:
1. **Interaktive Eingabe:** Nutzer beantworten 19 Fragen auf einer 5-stufigen Likert-Skala (von *Trifft gar nicht zu* bis *Trifft voll zu*) und geben demografische Daten an.
2. **Echtzeit-Inferenz:** Die Eingaben werden strukturiert, durch die vortrainierte Pipeline transformiert und direkt klassifiziert.
3. **Ergebnisdarstellung:** Ausgabe des prognostizierten Persönlichkeitstyps inklusive Profilbeschreibung.
---

## 🚀 2. Einrichtung (Setup)

Schritt-für-Schritt-Anleitung zur vollständigen Reproduktion des Projekts:

### 1. Repository klonen
```bash
git clone https://github.com/markant83/Personality_Profile_Predictor.git
cd Personality_Profile_Predictor
```

### 2. Daten beschaffen
1. Lade den Rohdatensatz über folgenden Google-Drive-Link herunter:
   👉 `https://drive.google.com/file/d/1Qnc2hC9szsKlHiutJIttwhnxphGffn-2/view?usp=drive_link`
2. Erstelle im Projektordner das Verzeichnis `data/` (sofern noch nicht vorhanden):
   ```bash
   mkdir data
   ```
3. Lege die heruntergeladene CSV-Datei direkt in diesem Ordner ab (z. B. als `data/data.csv`).

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

<details>
<summary><h3 style="display:inline;">7. Streamlit-App im web starten</h3></summary>
 
Entferne die Zeile `*.joblib` aus der `.gitignore`.
Führe im Terminal folgende Befehle aus:
```bash
git add *.joblib .gitignore
git commit -m "Add joblib model to repo"
git push
```
Die App kann anschließend im Browser unter `https://personality-profile-predictor-mb.streamlit.app` geöffnet werden.
</details>

<details>
<summary><h3 style="display:inline;">8. Modell wieder aus dem Repository entfernen</h3></summary>

Füge die Zeile `*.joblib` der `.gitignore` hinzu.

Führe im Terminal folgende Befehle aus:
```bash
git rm --cached *.joblib
git add .gitignore
git commit -m "Remove joblib model from repo"
git push
```
Das Model wird aus GitHub entfernt, liegt lokal aber weiterhin in deinem Ordner.
</details>

---

## 🛠 Tech-Stack

* **Sprache:** Python 3.10+
* **ML & Datenverarbeitung:** Scikit-Learn, Pandas, NumPy
* **UI & Deployment:** Streamlit, Joblib
* **Experiment-Tracking:** MLflow
