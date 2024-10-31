import numpy as np  # Importa la librería numpy para operaciones matemáticas y manejo de arreglos.
import matplotlib.pyplot as plt  # Importa la librería matplotlib para graficar.
import random

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons  # Define la cantidad de neuronas en la red.
        self.weights = np.zeros((num_neurons, num_neurons))  # Inicializa la matriz de pesos sinápticos con ceros.

    def train(self, patterns):# Entrena la red con una lista de patrones.
        for p in patterns:
            self.weights += np.outer(p, p) # Actualiza los pesos sinápticos utilizando la regla de Hebb: suma el producto exterior del patrón consigo mismo.
        np.fill_diagonal(self.weights, 0) # Establece a cero la diagonal principal de la matriz de pesos (sin autoconexiones).
        self.weights /= len(patterns) # Normaliza los pesos dividiéndolos entre la cantidad de patrones.

    def predict(self, input_pattern, max_iterations=200): # Predice o recupera un patrón a partir de un patrón de entrada.
        output_pattern = input_pattern.copy()  # Crea una copia del patrón de entrada para evitar modificar el original.
        for _ in range(max_iterations): # Actualiza el patrón a lo largo de un número máximo de iteraciones.
            indices = np.random.permutation(self.num_neurons)  # Genera una permutación aleatoria de índices para actualizar.
            for i in indices:
                net_input = np.dot(self.weights[i], output_pattern) # Calcula la entrada neta para la neurona i.
                output_pattern[i] = 1 if net_input >= 0 else -1 # Actualiza el estado de la neurona i según si la entrada neta es positiva o negativa.
        return output_pattern  # Devuelve el patrón recuperado después de las iteraciones.

def generate_letters():
    letter_patterns = []
    # Genera un patrón para la letra 'A' en una matriz de 10x10.
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

    # Genera un patrón para la letra 'B' en una matriz de 10x10.
    I_pattern = [
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1
    ]

    # Genera un patrón para la letra 'C' en una matriz de 10x10.
    C_pattern = [
        -1,  1,  1,  1,  1,  1,  1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1,  1,  1,  1,  1,  1,  1, -1, -1, -1
    ]
    
    letter_patterns.append(A_pattern)
    #letter_patterns.append(I_pattern)
    letter_patterns.append(C_pattern)

    return np.array(letter_patterns)

def generate_letter_pattern(letter):

    pattern = np.zeros((10, 10), dtype=int)

    if letter == 'D':
        # Representación de la letra 'D'
        pattern[:, 0] = 1
        pattern[0, :-1] = 1
        pattern[-1, :-1] = 1
        pattern[1:-1, -1] = 1
    
    elif letter == 'E':
        # Representación de la letra 'E'
        pattern[:, 0] = 1
        pattern[0, :] = 1
        pattern[5, :7] = 1
        pattern[-1, :] = 1
    
    elif letter == 'F':
        # Representación de la letra 'F'
        pattern[:, 0] = 1
        pattern[0, :] = 1
        pattern[5, :7] = 1
    
    else:
        raise ValueError("Letra no reconocida o no implementada. Usa letras distintas de 'A', 'B', 'C'.")

    return pattern.flatten()

# Generar 50 patrones de la letra A con ruido
def generate_noisy_patterns(base_pattern, num_patterns, noise_level):
    noisy_patterns = []
    for i in range(len(base_pattern)):
        for _ in range(num_patterns):
            noisy_pattern = base_pattern[i].copy()  # Crea una copia del patrón base (la letra 'A').
            noise = np.random.choice([0, 1], size=base_pattern[i].shape, p=[1 - noise_level, noise_level]) # Genera un vector de ruido aleatorio de tamaño igual al patrón base.
            noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern) # Aplica el ruido: donde el valor de ruido es 1, invierte el valor del patrón.
            noisy_patterns.append(noisy_pattern)  # Agrega el patrón ruidoso a la lista.
    return np.array(noisy_patterns)  # Devuelve un array con todos los patrones ruidosos.

# Crear la red de Hopfield
num_neurons = 100  # Define el número de neuronas, que es igual al tamaño del patrón (10x10 = 100).
hopfield_net = HopfieldNetwork(num_neurons)  # Crea una instancia de la red de Hopfield con 100 neuronas.

# Genera patrón random similar a la letra A para probar
def generated_random_pattern(base_pattern, noise_level):
    noisy_pattern = base_pattern.copy()  # Crea una copia del patrón base (letra 'A').
    noise = np.random.choice([0, 1], size=base_pattern.shape, p=[1 - noise_level, noise_level]) # Genera ruido aleatorio para modificar el patrón.
    noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern)  # Aplica el ruido.
    pattern = noisy_pattern.copy()  # Crea una copia del patrón ruidoso.
    return np.array(pattern)  # Devuelve el patrón con ruido.

# Generar el patrón de la letra A y los patrones con ruido
patterns = generate_letters()
noisy_patterns = generate_noisy_patterns(patterns, num_patterns=50, noise_level=0.1)  # Genera 50 versiones ruidosas de 'A, B y C'.

# Entrenar la red con los patrones ruidosos
hopfield_net.train(noisy_patterns)  # Entrena la red de Hopfield con los patrones ruidosos.

Pattern_Test = []
tamano = len(noisy_patterns)
num_rand = random.randint(0, tamano)

print("Cantidad de patrones base:", len(patterns))
print("Patron random: ", num_rand)
print("Tamaño: ", tamano)
Pattern_Test = noisy_patterns[num_rand].copy()

# Agarramos un pater de prueba
#test_pattern = generated_random_pattern(Pattern_Test, noise_level=0.01).flatten()

# Ejemplo de uso:
test_letter = 'F'  # Cambia esta letra por la que quieras probar (que no sea A, B o C)
test_pattern = generate_letter_pattern(test_letter)

# Mostrar el patrón inicial
plt.subplot(1, 2, 1)  # Crea un subplot (1 fila, 2 columnas, posición 1) para mostrar el patrón inicial.
plt.title("Patrón inicial")  # Establece el título del gráfico.
plt.imshow(test_pattern.reshape((10, 10)), cmap="gray")  # Muestra el patrón inicial (con ruido) como una imagen 10x10 en una escala de colores.

# Recuperar el patrón usando la red de Hopfield
output_pattern = hopfield_net.predict(test_pattern, max_iterations=200)  # Recupera el patrón usando la red de Hopfield.

# Mostrar el patrón recuperado
plt.subplot(1, 2, 2)  # Crea otro subplot (posición 2) para mostrar el patrón recuperado.
plt.title("Patrón recuperado")  # Establece el título del gráfico.
plt.imshow(output_pattern.reshape((10, 10)), cmap="gray")  # Muestra el patrón recuperado como una imagen 10x10.
plt.show()  # Muestra la ventana gráfica con los dos subplots.

