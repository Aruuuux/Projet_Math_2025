import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

# Charger les données digits
digits = load_digits()
X, y = digits.data, digits.target

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Partition des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de régression logistique
clf = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)

# Prédictions
test_preds = clf.predict(X_test)

# Évaluation
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Précision sur l'ensemble de test : {test_accuracy:.4f}")

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, test_preds)
ConfusionMatrixDisplay(conf_matrix, display_labels=clf.classes_).plot(cmap='viridis')
plt.title("Matrice de confusion")
plt.show()

# Affichage des prédictions pour quelques chiffres
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.axis('off')
    pred_label = test_preds[i]
    true_label = y_test[i]
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"Préd: {pred_label}\nVrai: {true_label}", color=color)
plt.tight_layout()
plt.show()
