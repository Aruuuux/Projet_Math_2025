import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from descente_stochastique import GradientDescent

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Limite les valeurs pour éviter le débordement
    return 1 / (1 + np.exp(-z))

def compute_gradient(weights, x_batch, y_batch):
    m = x_batch.shape[0]
    predictions = sigmoid(np.dot(x_batch, weights))
    errors = predictions - y_batch
    gradient = np.dot(x_batch.T, errors) / m
    return gradient

# Charger les données digits
digits = load_digits()
X, y = digits.data, digits.target

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ajout de biais (colonne de 1s pour l'interception)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Partition des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialisation des paramètres
num_classes = len(np.unique(y))
initial_weights = np.zeros((X_train.shape[1], num_classes))  # Poids pour chaque classe
learning_rate = 0.1
max_iterations = 500

def wrapped_gradient(weights, batch):
    x_batch = batch[:, :-1]  # Toutes les colonnes sauf la dernière
    y_batch = batch[:, -1].astype(int)  # La dernière colonne
    predictions = sigmoid(np.dot(x_batch, weights))  # Prédictions pour chaque classe
    one_hot = np.eye(num_classes)[y_batch]  # Convertir les labels en one-hot encoding
    errors = predictions - one_hot
    gradient = np.dot(x_batch.T, errors) / x_batch.shape[0]
    return gradient

data = np.hstack((X_train, y_train.reshape(-1, 1)))
gd = GradientDescent(gradient=wrapped_gradient, learning_rate=learning_rate, max_iterations=max_iterations, batch_size=32)

# Entraînement
weights = gd.descent(initial_weights, data)

# Prédictions (argmax sur les probabilités pour chaque classe)
train_preds = np.argmax(sigmoid(np.dot(X_train, weights)), axis=1)
test_preds = np.argmax(sigmoid(np.dot(X_test, weights)), axis=1)

# Évaluation
train_accuracy = np.mean(train_preds == y_train)
test_accuracy = np.mean(test_preds == y_test)
print(f"Précision sur l'ensemble de test (implémentation personnalisée) : {test_accuracy:.4f}")

# Matrice de confusion
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(y_test, test_preds):
    conf_matrix[true, pred] += 1

# Affichage de la matrice de confusion
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(conf_matrix, cmap='viridis')
plt.title("Confusion Matrix", pad=20)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.colorbar(cax)
plt.xticks(np.arange(num_classes), labels=np.arange(num_classes))
plt.yticks(np.arange(num_classes), labels=np.arange(num_classes))

# Ajouter les valeurs dans la matrice
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', 
                 color='black' if conf_matrix[i, j] > np.max(conf_matrix) / 2 else 'white', fontsize=12)

plt.show()

# Affichage des prédictions pour quelques chiffres
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i, 1:].reshape(8, 8), cmap='gray')
    ax.axis('off')
    pred_label = test_preds[i]
    true_label = y_test[i]
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"Préd: {pred_label}\nVrai: {true_label}", color=color)
plt.tight_layout()
plt.show()
