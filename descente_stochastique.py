import numpy as np

class GradientDescent:
    
    def __init__(self, gradient, learning_rate=0.01, max_iterations=1000, epsilon=1e-6, batch_size=1):
        
        """
        Initialise l'objet GradientDescent avec les paramètres nécessaires.

        Paramètres :
        - gradient : La fonction gradient de la fonction de coût.
        - learning_rate : Taux d'apprentissage initial pour la mise à jour des paramètres (par défaut à 0.01).
        - max_iterations : Nombre maximal d'itérations de l'algorithme de descente (par défaut à 1000).
        - epsilon : Niveau de tolérance pour l'arrêt de l'algorithme (par défaut à 1e-6).
        - batch_size : Taille du mini-lot pour la descente de gradient stochastique (par défaut à 1, pour le pur SGD).
        """
        
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.num_iterations = 0
        self.batch_size = batch_size

    def descent(self, initial_point, data=None):
        
        """
        Effectue l'algorithme de descente de gradient (classique ou stochastique).

        Paramètres :
        - initial_point : Le point de départ de l'algorithme.
        - data : Les données d'entraînement, nécessaires pour la descente stochastique.

        Retourne :
        - Le point optimal trouvé par l'algorithme.
        """
        
        current_point = initial_point
        stopped = False  # Flag pour contrôler l'arrêt anticipé
        
        for epoch in range(self.max_iterations):
            if data is not None:
                # Mélange des données au début de chaque époque
                np.random.shuffle(data)
        
                for i in range(0, len(data), self.batch_size):
                    mini_batch = data[i:min(i + self.batch_size, len(data))] # au cas où le dernier batch est plus petit
                    current_gradient = self.gradient(current_point, mini_batch)
                    
                    # Condition d'arrêt basée sur la norme du gradient avant mise à jour
                    if np.linalg.norm(current_gradient) < self.epsilon:
                        stopped = True
                        break
                    
                    # Mise à jour du point courant
                    current_point = self.update(current_point, current_gradient)
        
            else:
                # Utilisation du gradient classique (batch complet)
                current_gradient = self.gradient(current_point)
        
                # Condition d'arrêt basée sur la norme du gradient
                if np.linalg.norm(current_gradient) < self.epsilon:
                    stopped = True
                    break
        
                # Mise à jour du point courant
                current_point = self.update(current_point, current_gradient)
        
            # Sortie si la condition d'arrêt est remplie
            if stopped:
                break
        
        # Ajustement du nombre d'itérations 
        self.num_iterations = epoch + 1 if not stopped else epoch
        return current_point

    def update(self, point, gradient_value):
        
        """
        Met à jour le point en utilisant le gradient et le taux d'apprentissage.

        Paramètres :
        - point : Le point à mettre à jour.
        - gradient_value : Le gradient de la fonction de coût au point donné.

        Retourne :
        - Le nouveau point après la mise à jour.
        """
        
        return point - self.learning_rate * gradient_value
