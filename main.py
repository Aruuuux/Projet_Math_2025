import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from descente_stochastique import GradientDescent
import matplotlib.pyplot as plt

# Charger le dataset
digits = load_digits()
X = digits.data  # Images aplaties (64 features par image)
y = digits.target  # Labels (chiffres de 0 à 9)

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Partitionner les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction sigmoïde
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Entraîner un modèle pour chaque classe avec OvR
def train_one_vs_rest(X_train, y_train, num_classes, gradient_descent):
    """
    Entraîne un modèle pour chaque classe en utilisant l'approche One-vs-Rest.
    Retourne un tableau contenant les paramètres optimaux pour chaque classe.
    """
    models = []
    for class_label in range(num_classes):
        print(f"Entraînement pour la classe {class_label}...")
        # Créer un problème binaire : 1 pour la classe actuelle, 0 sinon
        y_binary = (y_train == class_label).astype(int)
        
        # Initialiser les paramètres
        initial_theta = np.zeros(X_train.shape[1])
        
        # Entraîner avec la descente de gradient
        theta_optimal = gradient_descent.descent(initial_theta, data=list(zip(X_train, y_binary)))
        models.append(theta_optimal)
    return np.array(models)

# Prédire les classes en utilisant les modèles OvR
def predict_one_vs_rest(X, models):
    """
    Prédire les classes des échantillons X en utilisant les modèles OvR.
    """
    scores = np.array([sigmoid(X @ theta) for theta in models])  # Probabilités pour chaque classe
    return np.argmax(scores, axis=0)  # Classe avec la probabilité maximale

# Initialiser la descente de gradient
gd = GradientDescent(
    gradient=lambda theta, data: np.mean([(x * (x @ theta - y)) for x, y in data], axis=0),
    learning_rate=0.01,
    max_iterations=1000,
    epsilon=1e-6,
    batch_size=10
)

# Nombre de classes (0 à 9)
num_classes = 10

# Entraîner les modèles
models = train_one_vs_rest(X_train, y_train, num_classes, gd)

# Faire des prédictions sur l'ensemble de test
y_pred = predict_one_vs_rest(X_test, models)

# Évaluer les performances
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision sur l'ensemble de test : {accuracy * 100:.2f}%")

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", conf_matrix)

# Visualisation des échantillons mal classés
def plot_misclassified_images(X, y_true, y_pred, n_images=10):
    """
    Affiche les images mal classées.
    """
    misclassified = np.where(y_true != y_pred)[0]
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(misclassified[:n_images]):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(X[idx].reshape(8, 8), cmap='gray')
        plt.title(f"Vrai: {y_true[idx]}\nPrédit: {y_pred[idx]}")
        plt.axis('off')
    plt.show()

# Afficher les 10 premières images mal classées
plot_misclassified_images(X_test, y_test, y_pred)
