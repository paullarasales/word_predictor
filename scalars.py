import numpy as np
import matplotlib.pyplot as plt

vector_a = np.array([2, 3])
vector_b = np.array([4, 5])

vector_sum = vector_a + vector_b
print(f"Vector Sum: {vector_sum}")

scalar = 2
scaled_vector_a = scalar * vector_a
scaled_vector_b = scalar * vector_b
print(f"Scaled Vector A: {scaled_vector_a}, Scaled Vector B: {scaled_vector_b}")

magnitude_a = np.linalg.norm(vector_a)
magnitude_b = np.linalg.norm(vector_b)

unit_vector_a = vector_a / magnitude_a
unit_vector_b = vector_b / magnitude_b

plt.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector A')
plt.quiver(0, 0, scaled_vector_a[0], scaled_vector_a[1], angles='xy', scale_units='xy', scale=1, color='pink', label='Vector A x 2')
plt.quiver(0, 0, vector_b[0], vector_b[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector B')
plt.quiver(0, 0, scaled_vector_b[0], scaled_vector_b[1], angles='xy', scale_units='xy', scale=1, color='c', label='Vector B x 2')
plt.quiver(0, 0, vector_sum[0], vector_sum[1], angles='xy', scale_units='xy', scale=1, color='g', label='A + B')

plt.xlim(0, 10)
plt.ylim(0, 10)

plt.grid()
plt.legend()
plt.show()
