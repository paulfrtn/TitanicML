import pandas as pd

# Lecture du fichier CSV
df = pd.read_csv('./Submission_titanic.csv')

# Conserver uniquement les colonnes 'PassengerId' et 'Survived'
df_filtered = df[['PassengerId', 'Survived']]

# Sauvegarde du fichier modifié
df_filtered.to_csv('fichier_modifié.csv', index=False)

print("Fichier modifié avec succès.")
