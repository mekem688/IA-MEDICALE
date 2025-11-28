import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# Charger ton dataset original une seule fois
df = pd.read_csv("diabetes_dataset.csv")
df = df.fillna(df.median(numeric_only=True))
df = df.fillna("Unknown")

# Encodage des colonnes catégorielles
cat_cols = df.select_dtypes(include=['object']).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Préparer les variables
X = df.drop(columns=["diabetes_risk_score", "diabetes_stage", "diagnosed_diabetes"])
y_reg = df["diabetes_risk_score"]
y_class = df[["diabetes_stage", "diagnosed_diabetes"]]

X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Entraîner les modèles une seule fois
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_train, y_reg_train)

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train, y_class_train)

# Sauvegarder modèle + encodeurs + colonnes
joblib.dump({
    "reg_model": reg_model,
    "clf_model": clf_model,
    "encoders": encoders,
    "cat_cols": list(cat_cols),
    "columns": list(X.columns)
}, "modele_diabete.joblib")

print("✅ Modèle entraîné et sauvegardé dans 'modele_diabete.joblib'")
