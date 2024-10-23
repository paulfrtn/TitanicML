# Imports
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

# Lecture de données
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

# Vérification des données
print("Train Dataset Shape:", df_train.shape)
print("Test Dataset Shape:", df_test.shape)

# Gestion des valeurs manquantes
# Remplacer les valeurs manquantes dans 'Age' par la médiane
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)

# Remplacer les valeurs manquantes dans 'Embarked' par la valeur la plus fréquente
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].mode()[0], inplace=True)

# Remplacer les valeurs manquantes dans 'Fare' (test set uniquement) par la médiane
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

# Colonnes à supprimer car inutiles pour la modélisation
df_train.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
df_test.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Assurer que les colonnes 'Sex' et 'Embarked' sont bien de type string
df_train['Sex'] = df_train['Sex'].astype(str)
df_train['Embarked'] = df_train['Embarked'].astype(str)
df_test['Sex'] = df_test['Sex'].astype(str)
df_test['Embarked'] = df_test['Embarked'].astype(str)

# Vérifier les valeurs manquantes dans ces colonnes
print(df_train[['Sex', 'Embarked']].isnull().sum())
print(df_test[['Sex', 'Embarked']].isnull().sum())

# Encodage des variables catégorielles
label_encoder_sex = LabelEncoder()
label_encoder_embarked = LabelEncoder()

# Encoder 'Sex' et 'Embarked' pour les deux jeux de données
df_train['Sex'] = label_encoder_sex.fit_transform(df_train['Sex'])
df_train['Embarked'] = label_encoder_embarked.fit_transform(df_train['Embarked'])
df_test['Sex'] = label_encoder_sex.transform(df_test['Sex'])
df_test['Embarked'] = label_encoder_embarked.transform(df_test['Embarked'])

# Séparation des features (X) et de la cible (Y)
X = df_train.drop('Survived', axis=1)  # Enlever la colonne cible pour X
Y = df_train['Survived']  # La colonne cible

# Division des données en jeu d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Vérifier la taille des jeux de données
print("Taille de X_train :", X_train.shape)
print("Taille de X_test :", X_test.shape)
print("Taille de Y_train :", Y_train.shape)
print("Taille de Y_test :", Y_test.shape)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle de Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

gradient_boosting = GradientBoostingClassifier(random_state=42)

grid_search_gb = GridSearchCV(estimator=gradient_boosting, param_grid=param_grid_gb,
                              cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

# Entraîner le modèle
grid_search_gb.fit(X_train_scaled, Y_train)

# Meilleur modèle
best_gradient_boosting = grid_search_gb.best_estimator_

# Prédiction sur le jeu de test
Y_pred_gb = best_gradient_boosting.predict(X_test_scaled)

# Calculer le score F1
f1_score_gb = metrics.f1_score(Y_test, Y_pred_gb, average='macro')
print(f'Macro F1 Score (Gradient Boosting): {f1_score_gb}')
