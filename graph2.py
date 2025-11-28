import pandas as pd 
import os
import time
from threading import Thread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import Canvas

# ===== Couleurs et styles =====
BG_TOP = "#00c6ff"
BG_BOTTOM = "#0072ff"
FRAME_COLOR = "#e0f7fa"
FG_COLOR = "#004d40"
BUTTON_COLOR = "#00796b"
BUTTON_FG = "#ffffff"
TITLE_FONT = ("Helvetica", 16, "bold")
LABEL_FONT = ("Helvetica", 11)

# ===== Traduction complète FR → EN =====
traduction_fr_en = {
    "homme": "Male",
    "femme": "Female",
    "asiatique": "Asian",
    "blanc": "White",
    "hispanique": "Hispanic",
    "noir": "Black",
    "autre": "Other",
    "collège": "Highschool",
    "lycée": "Highschool",
    "diplômé": "Graduate",
    "universitaire": "Graduate",
    "doctorat": "Doctorate",
    "faible": "Low",
    "moyen-faible": "Lower-Middle",
    "moyen": "Middle",
    "moyen-élevé": "Upper-Middle",
    "élevé": "High",
    "employé": "Employed",
    "chômeur": "Unemployed",
    "étudiant": "Student",
    "retraité": "Retired",
    "jamais": "Never",
    "ancien fumeur": "Former",
    "fumeur": "Current",
    "oui": "1",
    "non": "0",
    "pas de diabète": "No Diabetes",
    "pré-diabète": "Pre-Diabetes",
    "type 1": "Type 1",
    "type 2": "Type 2",
}

# Traduction inverse EN → FR pour l'affichage
traduction_en_fr = {v.lower(): k for k, v in traduction_fr_en.items()}

# ===== Labels en français =====
labels_francais = {
    "age": "Âge",
    "gender": "Genre",
    "ethnicity": "Ethnicité",
    "education_level": "Niveau d'éducation",
    "income_level": "Niveau de revenu",
    "employment_status": "Statut d'emploi",
    "smoking_status": "Statut fumeur",
    "alcohol_consumption_per_week": "Consommation d'alcool (verres/semaine)",
    "physical_activity_minutes_per_week": "Activité physique (min/semaine)",
    "diet_score": "Score diététique (0-10)",
    "sleep_hours_per_day": "Heures de sommeil par jour",
    "screen_time_hours_per_day": "Temps d'écran (heures/jour)",
    "family_history_diabetes": "Antécédents familiaux de diabète",
    "hypertension_history": "Antécédents d'hypertension",
    "cardiovascular_history": "Antécédents cardiovasculaires",
    "bmi": "IMC",
    "waist_to_hip_ratio": "Ratio taille/hanche",
    "systolic_bp": "Pression artérielle systolique",
    "diastolic_bp": "Pression artérielle diastolique",
    "heart_rate": "Fréquence cardiaque",
    "cholesterol_total": "Cholestérol total",
    "hdl_cholesterol": "Cholestérol HDL",
    "ldl_cholesterol": "Cholestérol LDL",
    "triglycerides": "Triglycérides",
    "glucose_fasting": "Glucose à jeun",
    "glucose_postprandial": "Glucose postprandial",
    "insulin_level": "Niveau d'insuline",
    "hba1c": "HbA1c (%)"
}

# ===== Options françaises pour dropdowns =====
options_francaises = {
    "gender": ["Homme", "Femme"],
    "ethnicity": ["Asiatique", "Blanc", "Hispanique", "Noir", "Autre"],
    "education_level": ["Collège", "Lycée", "Universitaire", "Doctorat"],
    "income_level": ["Faible", "Moyen-Faible", "Moyen", "Moyen-Élevé", "Élevé"],
    "employment_status": ["Employé", "Chômeur", "Étudiant", "Retraité"],
    "smoking_status": ["Jamais", "Ancien fumeur", "Fumeur"],
    "family_history_diabetes": ["Non", "Oui"],
    "hypertension_history": ["Non", "Oui"],
    "cardiovascular_history": ["Non", "Oui"]
}

# ===== Charger dataset =====
base_dir = os.path.dirname(os.path.abspath(__file__))
fichier = os.path.join(base_dir, "diabetes_dataset.csv")

try:
    df_dataset = pd.read_csv(fichier)
    df_dataset = df_dataset.fillna(df_dataset.median(numeric_only=True))
    df_dataset = df_dataset.fillna("Unknown")
except FileNotFoundError:
    messagebox.showerror("Erreur", "Le fichier 'diabetes_dataset.csv' est introuvable !")
    exit()

# ===== Interface principale =====
root = tk.Tk()
root.title("🩺 Prédiction du Diabète")
root.geometry("950x800")  # fenêtre initiale
root.minsize(800, 600)    # réduire
root.maxsize(1600, 1200)  # agrandir

# ===== Canvas + Scrollbar =====
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, bg=BG_TOP)
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg=FRAME_COLOR)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ===== Bouton prédiction en haut =====
pred_button = tk.Button(scrollable_frame, text="🔮 Prédire le risque de diabète", font=("Helvetica", 12, "bold"), 
                        bg=BUTTON_COLOR, fg=BUTTON_FG, relief="raised", padx=20, pady=10,
                        command=lambda: Thread(target=lambda: predict()).start())
pred_button.pack(pady=10)

# ===== Barre de progression =====
progress_label = tk.Label(scrollable_frame, text="Préparation de l'IA...", bg=FRAME_COLOR, fg=FG_COLOR, font=TITLE_FONT)
progress_label.pack(pady=5)
progress = ttk.Progressbar(scrollable_frame, orient='horizontal', length=400, mode='determinate')
progress.pack(pady=5)

# ===== Entraînement en arrière-plan =====
def train_models():
    global X, cat_cols, encoders, reg_model, clf_model
    df = df_dataset.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        progress['value'] = (i+1)/len(cat_cols)*50
        root.update()
        time.sleep(0.02)
    
    X = df.drop(columns=["diabetes_risk_score", "diabetes_stage", "diagnosed_diabetes"])
    y_reg = df["diabetes_risk_score"]
    y_class = df[["diabetes_stage", "diagnosed_diabetes"]]

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

    progress_label.config(text="Entraînement modèle régression...")
    root.update()
    reg_model = RandomForestRegressor(random_state=42)
    reg_model.fit(X_train, y_reg_train)

    progress_label.config(text="Entraînement modèle classification...")
    root.update()
    clf_model = RandomForestClassifier(random_state=42)
    clf_model.fit(X_train, y_class_train)

    progress['value'] = 100
    progress_label.config(text="✅ IA prête ! Vous pouvez remplir le formulaire.")

Thread(target=train_models).start()

# ===== Formulaire =====
entries = {}

sections = {
    "Informations personnelles": ["age", "gender", "ethnicity", "education_level", "income_level", "employment_status"],
    "Habitudes et style de vie": ["smoking_status", "alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day"],
    "Antécédents médicaux": ["family_history_diabetes", "hypertension_history", "cardiovascular_history", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides", "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"]
}

def create_field(parent, col):
    var = tk.StringVar()
    
    if col in options_francaises:
        vals = options_francaises[col]
        var.set(vals[0])
        widget = ttk.OptionMenu(parent, var, vals[0], *vals)
        widget.config(width=25)
    else:
        vals = df_dataset[col].dropna().unique().tolist()
        if len(vals) > 0:
            var.set(str(vals[0]))
        widget = ttk.Entry(parent, textvariable=var, width=28)
    
    return var, widget

# ===== Remplir le formulaire =====
for section, cols in sections.items():
    tk.Label(scrollable_frame, text=section, bg=FRAME_COLOR, fg=FG_COLOR, font=("Helvetica", 14, "bold")).pack(pady=5, anchor="w", padx=10)
    for col in cols:
        frame = tk.Frame(scrollable_frame, bg=FRAME_COLOR)
        frame.pack(fill="x", padx=10, pady=2)
        tk.Label(frame, text=f"{labels_francais.get(col, col)} :", bg=FRAME_COLOR, fg=FG_COLOR, font=LABEL_FONT, width=35, anchor="w").pack(side="left")
        var, widget = create_field(frame, col)
        widget.pack(side="left")
        entries[col] = var

# ===== Fonction prédiction =====
def predict():
    try:
        new_data = {}
        for col, widget in entries.items():
            val = widget.get().strip()
            
            val_lower = val.lower()
            if val_lower in traduction_fr_en:
                val = traduction_fr_en[val_lower]
            
            try:
                val = float(val)
            except:
                pass
            
            new_data[col] = val

        user_df = pd.DataFrame([new_data])

        for col in cat_cols:
            if col in user_df.columns:
                le = encoders[col]
                try:
                    user_df[col] = le.transform(user_df[col].astype(str))
                except ValueError:
                    user_df[col] = le.transform(["Unknown"])

        risk_pred = reg_model.predict(user_df)[0]
        stage_pred, diag_pred = clf_model.predict(user_df)[0]
        stage_label = encoders["diabetes_stage"].inverse_transform([int(stage_pred)])[0]
        
        stage_fr = traduction_en_fr.get(stage_label.lower(), stage_label)
        diagnostic_fr = "Oui" if diag_pred == 1 else "Non"
        
        messagebox.showinfo("📊 Résultat de la Prédiction",
                            f"Score de risque de diabète : {risk_pred:.2f}\n\n"
                            f"Stade estimé : {stage_fr.title()}\n\n"
                            f"Diagnostic probable : {diagnostic_fr}")
    except Exception as e:
        messagebox.showerror("❌ Erreur", f"Une erreur s'est produite :\n{str(e)}")

root.mainloop()

import pandas as pd 
import os
import time
from threading import Thread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import Canvas

# ===== Couleurs et styles =====
BG_TOP = "#00c6ff"
BG_BOTTOM = "#0072ff"
FRAME_COLOR = "#e0f7fa"
FG_COLOR = "#004d40"
BUTTON_COLOR = "#00796b"
BUTTON_FG = "#ffffff"
TITLE_FONT = ("Helvetica", 16, "bold")
LABEL_FONT = ("Helvetica", 11)

# ===== Traduction complète FR → EN =====
traduction_fr_en = {
    "homme": "Male",
    "femme": "Female",
    "asiatique": "Asian",
    "blanc": "White",
    "hispanique": "Hispanic",
    "noir": "Black",
    "autre": "Other",
    "collège": "Highschool",
    "lycée": "Highschool",
    "diplômé": "Graduate",
    "universitaire": "Graduate",
    "doctorat": "Doctorate",
    "faible": "Low",
    "moyen-faible": "Lower-Middle",
    "moyen": "Middle",
    "moyen-élevé": "Upper-Middle",
    "élevé": "High",
    "employé": "Employed",
    "chômeur": "Unemployed",
    "étudiant": "Student",
    "retraité": "Retired",
    "jamais": "Never",
    "ancien fumeur": "Former",
    "fumeur": "Current",
    "oui": "1",
    "non": "0",
    "pas de diabète": "No Diabetes",
    "pré-diabète": "Pre-Diabetes",
    "type 1": "Type 1",
    "type 2": "Type 2",
}

# Traduction inverse EN → FR pour l'affichage
traduction_en_fr = {v.lower(): k for k, v in traduction_fr_en.items()}

# ===== Labels en français =====
labels_francais = {
    "age": "Âge",
    "gender": "Genre",
    "ethnicity": "Ethnicité",
    "education_level": "Niveau d'éducation",
    "income_level": "Niveau de revenu",
    "employment_status": "Statut d'emploi",
    "smoking_status": "Statut fumeur",
    "alcohol_consumption_per_week": "Consommation d'alcool (verres/semaine)",
    "physical_activity_minutes_per_week": "Activité physique (min/semaine)",
    "diet_score": "Score diététique (0-10)",
    "sleep_hours_per_day": "Heures de sommeil par jour",
    "screen_time_hours_per_day": "Temps d'écran (heures/jour)",
    "family_history_diabetes": "Antécédents familiaux de diabète",
    "hypertension_history": "Antécédents d'hypertension",
    "cardiovascular_history": "Antécédents cardiovasculaires",
    "bmi": "IMC",
    "waist_to_hip_ratio": "Ratio taille/hanche",
    "systolic_bp": "Pression artérielle systolique",
    "diastolic_bp": "Pression artérielle diastolique",
    "heart_rate": "Fréquence cardiaque",
    "cholesterol_total": "Cholestérol total",
    "hdl_cholesterol": "Cholestérol HDL",
    "ldl_cholesterol": "Cholestérol LDL",
    "triglycerides": "Triglycérides",
    "glucose_fasting": "Glucose à jeun",
    "glucose_postprandial": "Glucose postprandial",
    "insulin_level": "Niveau d'insuline",
    "hba1c": "HbA1c (%)"
}

# ===== Options françaises pour dropdowns =====
options_francaises = {
    "gender": ["Homme", "Femme"],
    "ethnicity": ["Asiatique", "Blanc", "Hispanique", "Noir", "Autre"],
    "education_level": ["Collège", "Lycée", "Universitaire", "Doctorat"],
    "income_level": ["Faible", "Moyen-Faible", "Moyen", "Moyen-Élevé", "Élevé"],
    "employment_status": ["Employé", "Chômeur", "Étudiant", "Retraité"],
    "smoking_status": ["Jamais", "Ancien fumeur", "Fumeur"],
    "family_history_diabetes": ["Non", "Oui"],
    "hypertension_history": ["Non", "Oui"],
    "cardiovascular_history": ["Non", "Oui"]
}

# ===== Charger dataset =====
base_dir = os.path.dirname(os.path.abspath(__file__))
fichier = os.path.join(base_dir, "diabetes_dataset.csv")

try:
    df_dataset = pd.read_csv(fichier)
    df_dataset = df_dataset.fillna(df_dataset.median(numeric_only=True))
    df_dataset = df_dataset.fillna("Unknown")
except FileNotFoundError:
    messagebox.showerror("Erreur", "Le fichier 'diabetes_dataset.csv' est introuvable !")
    exit()

# ===== Interface principale =====
root = tk.Tk()
root.title("🩺 Prédiction du Diabète")
root.geometry("950x800")  # fenêtre initiale
root.minsize(800, 600)    # réduire
root.maxsize(1600, 1200)  # agrandir

# ===== Canvas + Scrollbar =====
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, bg=BG_TOP)
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg=FRAME_COLOR)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ===== Bouton prédiction en haut =====
pred_button = tk.Button(scrollable_frame, text="🔮 Prédire le risque de diabète", font=("Helvetica", 12, "bold"), 
                        bg=BUTTON_COLOR, fg=BUTTON_FG, relief="raised", padx=20, pady=10,
                        command=lambda: Thread(target=lambda: predict()).start())
pred_button.pack(pady=10)

# ===== Barre de progression =====
progress_label = tk.Label(scrollable_frame, text="Préparation de l'IA...", bg=FRAME_COLOR, fg=FG_COLOR, font=TITLE_FONT)
progress_label.pack(pady=5)
progress = ttk.Progressbar(scrollable_frame, orient='horizontal', length=400, mode='determinate')
progress.pack(pady=5)

# ===== Entraînement en arrière-plan =====
def train_models():
    global X, cat_cols, encoders, reg_model, clf_model
    df = df_dataset.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        progress['value'] = (i+1)/len(cat_cols)*50
        root.update()
        time.sleep(0.02)
    
    X = df.drop(columns=["diabetes_risk_score", "diabetes_stage", "diagnosed_diabetes"])
    y_reg = df["diabetes_risk_score"]
    y_class = df[["diabetes_stage", "diagnosed_diabetes"]]

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

    progress_label.config(text="Entraînement modèle régression...")
    root.update()
    reg_model = RandomForestRegressor(random_state=42)
    reg_model.fit(X_train, y_reg_train)

    progress_label.config(text="Entraînement modèle classification...")
    root.update()
    clf_model = RandomForestClassifier(random_state=42)
    clf_model.fit(X_train, y_class_train)

    progress['value'] = 100
    progress_label.config(text="✅ IA prête ! Vous pouvez remplir le formulaire.")

Thread(target=train_models).start()

# ===== Formulaire =====
entries = {}

sections = {
    "Informations personnelles": ["age", "gender", "ethnicity", "education_level", "income_level", "employment_status"],
    "Habitudes et style de vie": ["smoking_status", "alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day"],
    "Antécédents médicaux": ["family_history_diabetes", "hypertension_history", "cardiovascular_history", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides", "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"]
}

def create_field(parent, col):
    var = tk.StringVar()
    
    if col in options_francaises:
        vals = options_francaises[col]
        var.set(vals[0])
        widget = ttk.OptionMenu(parent, var, vals[0], *vals)
        widget.config(width=25)
    else:
        vals = df_dataset[col].dropna().unique().tolist()
        if len(vals) > 0:
            var.set(str(vals[0]))
        widget = ttk.Entry(parent, textvariable=var, width=28)
    
    return var, widget

# ===== Remplir le formulaire =====
for section, cols in sections.items():
    tk.Label(scrollable_frame, text=section, bg=FRAME_COLOR, fg=FG_COLOR, font=("Helvetica", 14, "bold")).pack(pady=5, anchor="w", padx=10)
    for col in cols:
        frame = tk.Frame(scrollable_frame, bg=FRAME_COLOR)
        frame.pack(fill="x", padx=10, pady=2)
        tk.Label(frame, text=f"{labels_francais.get(col, col)} :", bg=FRAME_COLOR, fg=FG_COLOR, font=LABEL_FONT, width=35, anchor="w").pack(side="left")
        var, widget = create_field(frame, col)
        widget.pack(side="left")
        entries[col] = var

# ===== Fonction prédiction =====
def predict():
    try:
        new_data = {}
        for col, widget in entries.items():
            val = widget.get().strip()
            
            val_lower = val.lower()
            if val_lower in traduction_fr_en:
                val = traduction_fr_en[val_lower]
            
            try:
                val = float(val)
            except:
                pass
            
            new_data[col] = val

        user_df = pd.DataFrame([new_data])

        for col in cat_cols:
            if col in user_df.columns:
                le = encoders[col]
                try:
                    user_df[col] = le.transform(user_df[col].astype(str))
                except ValueError:
                    user_df[col] = le.transform(["Unknown"])

        risk_pred = reg_model.predict(user_df)[0]
        stage_pred, diag_pred = clf_model.predict(user_df)[0]
        stage_label = encoders["diabetes_stage"].inverse_transform([int(stage_pred)])[0]
        
        stage_fr = traduction_en_fr.get(stage_label.lower(), stage_label)
        diagnostic_fr = "Oui" if diag_pred == 1 else "Non"
        
        messagebox.showinfo("📊 Résultat de la Prédiction",
                            f"Score de risque de diabète : {risk_pred:.2f}\n\n"
                            f"Stade estimé : {stage_fr.title()}\n\n"
                            f"Diagnostic probable : {diagnostic_fr}")
    except Exception as e:
        messagebox.showerror("❌ Erreur", f"Une erreur s'est produite :\n{str(e)}")

root.mainloop()