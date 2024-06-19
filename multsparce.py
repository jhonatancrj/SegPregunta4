import numpy as np
import scipy.sparse as sp

# Crear matrices dispersas aleatorias con más de 1000 filas y columnas
rows, cols = 1000, 1000
density = 0.01  # Densidad de los elementos no nulos

# Generar matrices dispersas aleatorias
matrix1 = sp.random(rows, cols, density=density, format='csr')
matrix2 = sp.random(rows, cols, density=density, format='csr')

# Multiplicar las matrices dispersas
result_matrix = matrix1.dot(matrix2)

# Convertir las matrices dispersas a formato denso para imprimir
dense_matrix1 = matrix1.toarray()
dense_matrix2 = matrix2.toarray()
dense_result_matrix = result_matrix.toarray()

# Mostrar algunas propiedades del resultado
print("Forma de la matriz resultante: ", result_matrix.shape)
print("Número de elementos distintos de cero en la matriz resultante: ", result_matrix.nnz)
print("Densidad de la matriz resultante: ", result_matrix.nnz / (rows * cols))

# Imprimir una parte de las matrices (primeras 30 filas y columnas)
print("Matriz 1 (primeras 30 filas and columnas):")
print(dense_matrix1[:30, :30])

print("Matriz 2 (primeras 30 filas and columnas):")
print(dense_matrix2[:30, :30])

print("Matriz Resultante (primeras 30 filas and columnas):")
print(dense_result_matrix[:30, :30])
