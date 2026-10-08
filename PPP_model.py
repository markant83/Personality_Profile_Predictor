import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------------------
# 1. Daten laden & bereinigen
# -------------------------------------------------------------
"""Lädt das Rohdaten-CSV, bereinigt Eingabefehler im Alter und validiert Skalenwerte."""
print("=== 1. DATEN LADEN & BEREINIGEN ===\n")
df = pd.read_csv("data/data.csv")
initial_rows = len(df)

# Tracking für betroffene Zeilen-Indizes
corrupted_indices = set()

# 1.1 Altersbereinigung
is_birth_year = df["age"].between(1920, 2026)
birth_year_indices = set(df[is_birth_year].index)

if is_birth_year.any():
    print(f"-> {is_birth_year.sum()} Geburtsjahre in 'age' erkannt und umgerechnet.")
    df.loc[is_birth_year, "age"] = 2026 - df.loc[is_birth_year, "age"]

invalid_age = df["age"] > 120
count_invalid_age = invalid_age.sum()
if count_invalid_age > 0:
    corrupted_indices.update(df[invalid_age].index)
    print(f"-> {count_invalid_age} Werte mit Alter > 120 auf NaN gesetzt.")
    df.loc[invalid_age, "age"] = np.nan

# 1.2 Likert-Skalen prüfen (Zulässiger Wertebereich: 1 bis 5)
likert_cols = [
    c
    for c in df.select_dtypes(include=["int64", "float64"]).columns
    if c not in ["age", "target"]
]

total_invalid_likert_values = 0
likert_corrupted_indices = set()

for col in likert_cols:
    invalid_mask = ~df[col].between(1, 5) & df[col].notna()
    count = invalid_mask.sum()
    if count > 0:
        total_invalid_likert_values += count
        bad_idx = df[invalid_mask].index
        likert_corrupted_indices.update(bad_idx)
        corrupted_indices.update(bad_idx)
        df.loc[invalid_mask, col] = np.nan

print(f"-> {total_invalid_likert_values} unzulässige Likert-Werte in {len(likert_corrupted_indices)} Zeilen gefunden und auf NaN gesetzt.")

# 1.3 Kategoriale Merkmale & Fehlwerte prüfen
cat_summary_cols = ["gender", "hand"]
print("\n-> Fehlwerte in kategorialen Spalten (werden in Pipeline via Modus imputiert):")
for col in cat_summary_cols:
    n_missing = df[col].isna().sum()
    pct_missing = (n_missing / initial_rows) * 100
    print(f"   - '{col}': {n_missing} fehlend ({pct_missing:.2f}%) | Häufigster Wert: {df[col].mode()[0]}")

# 1.4 Verteilung der Zielvariable (Target)
print("\n-> Verteilung der Zielklassen ('target'):")
target_dist = df["target"].value_counts(normalize=True) * 100
for label, pct in target_dist.items():
    print(f"   - {label:<16}: {pct:5.2f}% ({df['target'].value_counts()[label]} Fälle)")

# Zusammenfassung
modified_rows = birth_year_indices | corrupted_indices
print(f"\n-> Bereinigung abgeschlossen: {len(modified_rows)} Zeilen ({(len(modified_rows)/initial_rows)*100:.2f}%) bereinigt:")
print(f"   - {len(birth_year_indices)} Zeilen: Geburtsjahr in Alter umgerechnet")
print(f"   - {count_invalid_age} Zeilen: Alter > 120 auf NaN gesetzt")
print(f"   - {len(likert_corrupted_indices)} Zeilen: unzulässige Likert-Werte auf NaN gesetzt")
print()

# -------------------------------------------------------------
# 2. Multikollinearitätsanalyse & Feature-Selektion
# -------------------------------------------------------------
"""Identifiziert redundante numerische Features mit Pearson-Korrelation |r| > 0.85 und entfernt diese."""
print("=== 2. MULTIKOLLINEARITÄT PRÜFEN & FEATURES BEREINIGEN ===\n")
numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.drop("target", errors="ignore")
corr_matrix = df[numeric_features].corr().abs()

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
cols_to_drop = [
    column for column in upper_tri.columns if any(upper_tri[column] > 0.85)
]

# Höchste Korrelation im Datensatz ermitteln
max_corr = upper_tri.max().max()
col2 = upper_tri.max().idxmax()
col1 = upper_tri[col2].idxmax()

print(f"-> Höchste Korrelation im Datensatz: {col1} & {col2} mit r = {max_corr:.2f}")

if cols_to_drop:
    print(f"Gefundene redundante Spalten (|r| > 0.85): {cols_to_drop}")
    for col in cols_to_drop:
        correlated_with = upper_tri.index[upper_tri[col] > 0.85].tolist()
        print(f"  - '{col}' wird entfernt (hohe Korrelation mit: {correlated_with})")
    df = df.drop(columns=cols_to_drop)
else:
    print("-> Keine redundanten Features (|r| > 0.85) gefunden.")
print()

# -------------------------------------------------------------
# 3. Datensplit & Vorbereitungs-Pipeline
# -------------------------------------------------------------
"""Trennt Merkmale von der Zielvariable und erstellt Vorverarbeitungs-Pipelines für numerische und kategoriale Daten."""
print("=== 3. DATEN SPLITTEN & PIPELINE KONFIGURIEREN ===\n")
X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Trainingsdaten: {X_train.shape[0]} Zeilen | Testdaten: {X_test.shape[0]} Zeilen")
print(f"Features: {len(num_cols)} numerisch, {len(cat_cols)} kategorial\n")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ]
)

# -------------------------------------------------------------
# 4. Modellvergleich (5-Fold Cross-Validation)
# -------------------------------------------------------------
"""Vergleicht Basis-Klassifikatoren anhand des makro-gemittelten F1-Scores zur Auswahl des besten Algorithmus."""
print("=== 4. MODELLVERGLEICH (5-FOLD CV) ===\n")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

scores_dict = {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
    score = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1
    ).mean()
    scores_dict[name] = score
    print(f"-> {name:<22}: F1-Macro = {score:.4f}")

best_name = max(scores_dict, key=scores_dict.get)
print(f"\nBestes Modell: {best_name} ({scores_dict[best_name]:.4f})\n")

# -------------------------------------------------------------
# 5. Hyperparameter-Tuning (GridSearchCV)
# -------------------------------------------------------------
"""Führt eine vollständige Rastersuche über alle 27 Hyperparameter-Kombinationen durch."""
from sklearn.model_selection import GridSearchCV

print("=== 5. HYPERPARAMETER-TUNING (GRID SEARCH) ===\n")
param_grid = {
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__min_samples_leaf": [10, 20, 40],
}

full_pipe = Pipeline(
    [("preprocessor", preprocessor), ("clf", models[best_name])]
)

# 27 Kombinationen x 3 Folds = 81 Trainingsdurchläufe
search = GridSearchCV(
    full_pipe,
    param_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
)

print(f"Starte GridSearchCV (teste alle 27 Kombinationen)...")
search.fit(X_train, y_train)
best_model = search.best_estimator_

print(f"Bester CV-Score:  {search.best_score_:.4f}")
print(f"Beste Parameter:  {search.best_params_}\n")

# -------------------------------------------------------------
# 6. Experiment Tracking & Modell-Export
# -------------------------------------------------------------
"""Protokolliert Parameter und Metriken in MLflow und speichert die trainierte Pipeline als joblib-Datei."""
print("=== 6. EXPERIMENT TRACKING & SPEICHERN ===\n")
test_score = best_model.score(X_test, y_test)
mlflow.set_experiment("personality_classification")

with mlflow.start_run():
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("test_acc", test_score)
    mlflow.sklearn.log_model(
        sk_model=best_model, name="model", serialization_format="cloudpickle"
    )

joblib.dump(best_model, "best_personality_model.joblib")
print(f"Test-Set Genauigkeit: {test_score:.4f}")
print('Modell erfolgreich als "best_personality_model.joblib" exportiert.\n')



