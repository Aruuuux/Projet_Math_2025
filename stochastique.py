import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from descente_stochastique import GradientDescent
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def sigmoid(z):
    z = np.clip(z, -500, 500)  # Limite les valeurs pour éviter le débordement
    return 1 / (1 + np.exp(-z))


# Charger les données digits
digits = load_digits()
X, y = digits.data, digits.target

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ajout de biais (colonne de 1s pour l'interception)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Partition des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation des paramètres
num_classes = len(np.unique(y))
initial_weights = np.zeros((X_train.shape[1], num_classes))  # Poids pour chaque classe
learning_rate = 0.01
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
print(data)
gd = GradientDescent(gradient=wrapped_gradient, learning_rate=learning_rate, max_iterations=max_iterations, batch_size=1)

# Entraînement
weights = gd.descent(initial_weights, data)

# Prédictions (argmax sur les probabilités pour chaque classe)
test_preds = np.argmax(sigmoid(np.dot(X_test, weights)), axis=1)

# Évaluation
test_accuracy = np.mean(test_preds == y_test)
print(f"Précision sur l'ensemble de test (implémentation personnalisée) : {test_accuracy:.4f}")

conf_matrix = confusion_matrix(y_test, test_preds)
ConfusionMatrixDisplay(conf_matrix).plot(cmap='viridis')
plt.title("Matrice de confusion")
plt.show()

# Affichage des prédictions pour quelques chiffres
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i, 1:].reshape(8, 8), cmap='gray')
    ax.axis('off')
    pred_label = test_preds[i]
    true_label = y_test[i]
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"Préd: {pred_label}\nVrai: {true_label}", color=color)
plt.tight_layout()
plt.show()
