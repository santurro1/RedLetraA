import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def predict(self, input_pattern, max_iterations=200):
        output_pattern = input_pattern.copy()
        for _ in range(max_iterations):
            indices = np.random.permutation(self.num_neurons)  # Actualización en orden aleatorio
            for i in indices:
                net_input = np.dot(self.weights[i], output_pattern)
                output_pattern[i] = 1 if net_input >= 0 else -1
        return output_pattern

def generate_letter_A():
    """Genera un patrón para la letra 'A' en una matriz de 10x10."""
    A_pattern = [
        -1, -1,  1,  1,  1,  1,  1,  1, -1, -1,
        -1,  1, -1, -1, -1, -1, -1, -1,  1, -1,
        1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
        1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
        1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
        1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
        1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
        1, -1, -1, -1, -1, -1, -1, -1, -1,  1
    ]
    return np.array(A_pattern)

# Generar 50 patrones de la letra A con ruido
def generate_noisy_patterns(base_pattern, num_patterns=50, noise_level=0.1):
    noisy_patterns = []
    for _ in range(num_patterns):
        noisy_pattern = base_pattern.copy()
        noise = np.random.choice([0, 1], size=base_pattern.shape, p=[1 - noise_level, noise_level])
        noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern)
        noisy_patterns.append(noisy_pattern)
    return np.array(noisy_patterns)

# Crear la red de Hopfield
num_neurons = 100
hopfield_net = HopfieldNetwork(num_neurons)

# Generar el patrón de la letra A y los patrones con ruido
A_pattern = generate_letter_A()
noisy_patterns = generate_noisy_patterns(A_pattern, num_patterns=50, noise_level=0.05) #Caso positivo
#noisy_patterns = generate_noisy_patterns(A_pattern, num_patterns=50, noise_level=0.4) #forzar falla
#noisy_patterns = generate_noisy_patterns(A_pattern, num_patterns=200, noise_level=0.1) #forzar falla

# Entrenar la red con los patrones ruidosos
hopfield_net.train(noisy_patterns)

# Tomar uno de los patrones de entrenamiento con ruido
test_pattern = noisy_patterns[0].copy()

# Mostrar el patrón inicial
plt.subplot(1, 2, 1)
plt.title("Patrón inicial")
plt.imshow(test_pattern.reshape((10, 10)), cmap="gray")

# Recuperar el patrón usando la red de Hopfield
output_pattern = hopfield_net.predict(test_pattern, max_iterations=200)

# Mostrar el patrón recuperado
plt.subplot(1, 2, 2)
plt.title("Patrón recuperado")
plt.imshow(output_pattern.reshape((10, 10)), cmap="gray")
plt.show()


