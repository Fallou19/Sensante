import pandas as pd
import numpy as np
# Charger le dataset
df = pd. read_csv (" data / patients_dakar . csv ")
# Verifier les dimensions
print (f" Dataset : {df. shape [0]} patients , {df. shape [1]} colonnes ")
print (f"\ nColonnes : { list (df. columns )}")
print (f"\ nDiagnostics :\n{df[' diagnostic ']. value_counts ()}")

from sklearn . preprocessing import LabelEncoder
# Encoder les variables categoriques en nombres
# Le modele ne comprend que des nombres !
le_sexe = LabelEncoder ()
le_region = LabelEncoder ()
df[' sexe_encoded '] = le_sexe . fit_transform (df['sexe '])
df[' region_encoded '] = le_region . fit_transform (df['region '])
# Definir les features (X) et la cible (y)
feature_cols = ['age ', ' sexe_encoded ', ' temperature ', ' tension_sys ',
'toux ', 'fatigue ', 'maux_tete ', ' region_encoded ']
X = df[ feature_cols ]
y = df['diagnostic ']
print (f" Features : {X. shape }") # (500 , 8)
print (f" Cible : {y. shape }") # (500 ,)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% pour le test
    random_state=42,   # reproductibilité
    stratify=y           # proportions respectées
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")

from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,   # 100 arbres de décision
    random_state=42
)

# Entraîner sur les données d'entraînement
model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre d'arbres  : {model.n_estimators}")
print(f"Nombre features  : {model.n_features_in_}")
print(f"Classes          : {list(model.classes_)}")

# Prédire sur les données de test
y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction': y_pred[:10]
})
print(comparison)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)
print("
Rapport de classification :")
print(classification_report(y_test, y_pred))

# Visualiser (optionnel)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=model.classes_,
           yticklabels=model.classes_)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.show()

import joblib
import os

# Créer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sérialiser le modèle
joblib.dump(model, "models/model.pkl")

# Vérifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"Modèle sauvegardé : models/model.pkl")
print(f"Taille : {size/1024:.1f} Ko")

# Sauvegarder les encodeurs et la liste des features
joblib.dump(le_sexe,    "models/encoder_sexe.pkl")
joblib.dump(le_region,  "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés.")

# Charger le modèle DEPUIS LE FICHIER
model_loaded      = joblib.load("models/model.pkl")
le_sexe_loaded    = joblib.load("models/encoder_sexe.pkl")
le_region_loaded  = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")