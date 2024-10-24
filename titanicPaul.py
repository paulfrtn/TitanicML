#%%
# Imports des bibliothèques
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# Imports des données
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

# Valeurs manquantes
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Age'].fillna(df_train['Age'].median(), inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Fare'].fillna(df_train['Fare'].median(), inplace=True)

# On supprime les colonnes inutiles
columns_to_drop = ['Cabin', 'Ticket', 'Name']
df_train.drop(columns_to_drop, axis=1, inplace=True)
df_test.drop(columns_to_drop, axis=1, inplace=True)

# On met des colonnes catégorielles en string
categorical_columns = ['Sex', 'Embarked']
for col in categorical_columns:
    df_train[col] = df_train[col].astype(str)
    df_test[col] = df_test[col].astype(str)

label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df_train[col] = label_encoders[col].fit_transform(df_train[col])
    df_test[col] = label_encoders[col].transform(df_test[col])

# Séparation des features et de la target
X = df_train.drop('Survived', axis=1)
Y = df_train['Survived']

# On sépare les données en train et validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Dimensionnement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
df_test_scaled = scaler.transform(df_test)

param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

# Entraînement du model avec GridSearchCV
gradient_boosting = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(
    estimator=gradient_boosting,
    param_grid=param_grid_gb,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

# On entraîne le modèle
grid_search_gb.fit(X_train_scaled, Y_train)

# On récupère le meilleur modèle
best_gradient_boosting = grid_search_gb.best_estimator_

# On affiche les meilleurs paramètres
Y_test_pred = best_gradient_boosting.predict(X_test_scaled)
f1_score_test = metrics.f1_score(Y_test, Y_test_pred, average='macro')
print(f'Validation Macro F1 Score: {f1_score_test}')

# Prédiction sur le test set
test_predictions = best_gradient_boosting.predict(df_test_scaled)

# Créer un DataFrame avec les prédictions
submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': test_predictions
})

# Exporter en CSV
submission.to_csv('./data/submission.csv', index=False)
print("Fichier de soumission créé avec succès !")

# Vérifier les premières lignes du fichier
print("\nAperçu des premières lignes :")
print(submission.head())
#%% md
