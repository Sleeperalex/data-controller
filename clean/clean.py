import pandas as pd

def fill_missing_values(data : pd.DataFrame):
    # Parcourir chaque colonne du DataFrame
    for col in data.columns:
        # Vérifier si la colonne est numérique
        if data[col].dtype in ['int64', 'float64']:
            # Remplir les valeurs manquantes avec la moyenne pour les colonnes numériques
            data[col] = data[col].fillna(data[col].mean())
        # Vérifier si la colonne est de type texte (objet)
        elif data[col].dtype == 'object':
            # Remplir les valeurs manquantes avec 'Unknown' pour les colonnes de texte
            data[col] = data[col].fillna('Unknown')
    return data