import numpy as np  # Importa la librería numpy para operaciones matemáticas y manejo de arreglos.
import matplotlib.pyplot as plt  # Importa la librería matplotlib para graficar.
import random  # Importa la librería random para generar números aleatorios.

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons  # Define la cantidad de neuronas en la red.
        self.weights = np.zeros((num_neurons, num_neurons))  # Inicializa la matriz de pesos sinápticos con ceros.

    def train(self, patterns):  # Entrena la red con una lista de patrones.
        for p in patterns:
            self.weights += np.outer(p, p)  # Actualiza los pesos utilizando la regla de Hebb.
        np.fill_diagonal(self.weights, 0)  # Establece a cero la diagonal principal (sin autoconexiones).
        self.weights /= len(patterns)  # Normaliza los pesos dividiéndolos entre la cantidad de patrones.

    def predict(self, input_pattern, max_iterations=200):  # Predice o recupera un patrón a partir de un patrón de entrada.
        output_pattern = input_pattern.copy()  # Crea una copia del patrón de entrada.
        for _ in range(max_iterations):  # Actualiza el patrón a lo largo de un número máximo de iteraciones.
            indices = np.random.permutation(self.num_neurons)  # Genera una permutación aleatoria de índices.
            for i in indices:
                net_input = np.dot(self.weights[i], output_pattern)  # Calcula la entrada neta para la neurona i.
                output_pattern[i] = 1 if net_input >= 0 else -1  # Actualiza el estado de la neurona.
        return output_pattern  # Devuelve el patrón recuperado.

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
    letter_patterns.append(C_pattern)

    return np.array(letter_patterns)  # Devuelve todos los patrones de letras como un array.

def generate_letter_pattern(letter):
    pattern = np.zeros((10, 10), dtype=int)  # Inicializa un patrón de 10x10 con ceros.

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
        raise ValueError("Letra no reconocida o no implementada. Usa letras 'D', 'E' o 'F'.")

    return pattern.flatten()  # Devuelve el patrón como un vector.

# Generar patrones ruidosos a partir de un patrón base.
def generate_noisy_patterns(base_pattern, num_patterns, noise_level):
    noisy_patterns = []  # Lista para almacenar patrones ruidosos.
    for i in range(len(base_pattern)):
        for _ in range(num_patterns):
            noisy_pattern = base_pattern[i].copy()  # Crea una copia del patrón base.
            noise = np.random.choice([0, 1], size=base_pattern[i].shape, p=[1 - noise_level, noise_level])  # Genera ruido.
            noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern)  # Aplica el ruido.
            noisy_patterns.append(noisy_pattern)  # Agrega el patrón ruidoso a la lista.
    return np.array(noisy_patterns)  # Devuelve un array con todos los patrones ruidosos.

# Crear la red de Hopfield
num_neurons = 100  # Define el número de neuronas.
hopfield_net = HopfieldNetwork(num_neurons)  # Crea una instancia de la red de Hopfield.

# Genera patrón random similar a la letra A para probar
def generated_random_pattern(base_pattern, noise_level):
    noisy_pattern = base_pattern.copy()  # Crea una copia del patrón base (letra 'A').
    noise = np.random.choice([0, 1], size=base_pattern.shape, p=[1 - noise_level, noise_level]) # Genera ruido aleatorio para modificar el patrón.
    noisy_pattern = np.where(noise == 1, -noisy_pattern, noisy_pattern)  # Aplica el ruido.
    pattern = noisy_pattern.copy()  # Crea una copia del patrón ruidoso.
    return np.array(pattern)  # Devuelve el patrón con ruido.

# Generar el patrón de las letras y los patrones ruidosos
patterns = generate_letters()
noisy_patterns = generate_noisy_patterns(patterns, num_patterns=50, noise_level=0.1)  # Genera 50 patrones ruidosos.

# Entrenar la red con los patrones ruidosos
hopfield_net.train(noisy_patterns)  # Entrena la red de Hopfield.

start = 's'  # Variable de control para el bucle principal.

while start == 's':
    print("Desea probar con un patrón random o una letra específica? (1-2)")
    eleccion = int(input("Ingresa un número: "))  # Solicita la elección al usuario.

    if eleccion == 1:
        Pattern_Test = []
        tamano = len(noisy_patterns)
        num_rand = random.randint(0, tamano - 1)  # Selecciona un índice aleatorio de los patrones ruidosos.

        print("Cantidad de patrones base:", len(patterns))
        print("Patrón random: ", num_rand)
        print("Tamaño: ", tamano)
        Pattern_Test = noisy_patterns[num_rand].copy()  # Selecciona un patrón ruidoso aleatorio.
        test_pattern = generated_random_pattern(Pattern_Test, noise_level=0.01).flatten()  # Genera un patrón ruidoso.

    else:
        print("Probemos con una letra ('D', 'E', 'F')")
        cadena = input("Ingresa una letra de las de arriba: ")  # Solicita la letra al usuario.
        test_pattern = generate_letter_pattern(cadena)  # Genera el patrón de la letra.

    # Mostrar el patrón inicial
    plt.subplot(1, 2, 1)  # Crea un subplot para mostrar el patrón original.
    plt.title("Patrón original")
    plt.imshow(test_pattern.reshape(10, 10), cmap='gray')  # Muestra el patrón original.

    # Realizar la predicción
    recovered_pattern = hopfield_net.predict(test_pattern)  # Predice el patrón utilizando la red.

    # Mostrar el patrón recuperado
    plt.subplot(1, 2, 2)  # Crea otro subplot para mostrar el patrón recuperado.
    plt.title("Patrón recuperado")
    plt.imshow(recovered_pattern.reshape(10, 10), cmap='gray')  # Muestra el patrón recuperado.

    plt.show()  # Muestra ambas imágenes.

    start = input("Presiona 's' para continuar o cualquier otra tecla para salir: ")  # Pregunta si desea continuar.
