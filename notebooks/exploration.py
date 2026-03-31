"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et Structure Projet
"""
import pandas as pd

# ===== CHARGER LES DONNEES =====
df = pd.read_csv("data/patients_dakar.csv")

# ===== PREMIERS APERCUS =====
print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

# Dimensions du dataset
print("\nNombre de patients : " + str(len(df)))
print("Nombre de colonnes : " + str(df.shape[1]))
print("Colonnes : " + str(list(df.columns)))

# Apercu des 5 premieres lignes
print("\n--- 5 premiers patients ---")
print(df.head())

# ===== STATISTIQUES DE BASE =====
print("\n--- Statistiques descriptives ---")
print(df.describe().round(2))

# ===== REPARTITION DES DIAGNOSTICS =====
print("\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print("  " + str(diag) + " : " + str(count) + " patients (" + str(round(pct, 1)) + "%)")

# ===== REPARTITION PAR REGION =====
print("\n--- Repartition par region (top 5) ---")
region_counts = df["region"].value_counts().head(5)
for region, count in region_counts.items():
    print("  " + str(region) + " : " + str(count) + " patients")

# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
print("\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print("  " + str(diag) + " : " + str(round(temp, 1)) + "C")

print("\n" + "=" * 50)
print("Exploration terminee !")
print("Prochain lab : entrainer un modele ML")
print("=" * 50)