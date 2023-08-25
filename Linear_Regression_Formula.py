# this code given a XY dataset shows the line of best fit and the gamma of errors
# using a known formula of means

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from CSV (replace 'path/to/your/file.csv' with actual path)
csv_path = 'data.csv'
data = pd.read_csv(csv_path)

# Separate data into X and y
x = data[['x']].values
y = data[['y']].values


# Define function to calculate total error
def calculate_total_error(y, m, x, b, last_min, best_m, best_b):
    answer = np.mean((y - (m * x + b)) ** 2)
    if answer < last_min:
        last_min = answer
        best_m = m
        best_b = b
    return answer, last_min, best_m, best_b


# Create vectors for m and b
m_values = np.arange(-5, 5, 1, dtype=np.float64)
b_values = np.arange(-5, 5, 1, dtype=np.float64)

# Calculate total error for each combination of m and b
total_errors = np.empty((len(m_values), len(b_values)))
best_m = None
best_b = None
min_error = float('inf')

for i, m in enumerate(m_values):
    for j, b in enumerate(b_values):
        error, min_error, best_m, best_b = calculate_total_error(y, m, x, b, min_error, best_m, best_b)
        total_errors[i, j] = error

matrix = np.column_stack((m_values, b_values, total_errors))

# Create a meshgrid for m and b
M, B = np.meshgrid(m_values, b_values)

# Create the 3D plot
fig = plt.figure("INDEX PLOT")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, B, total_errors, cmap='viridis')
ax.scatter(best_m, best_b, color='red', s=50, label='Minimum Error Point')
ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('Total Error')

ax.set_title('3D Plot of Total Error')

plt.show()

# Create a 2D plot
fig = plt.figure("Best Fit Line Plot")
plt.scatter(x, y)
best_fit_line = best_m * x + best_b
plt.plot(x, best_fit_line, color='red', label='Best Fit Line')
# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple 2D Plot')
# Show the plot
plt.show()
print("best m:", best_m)
print("best b:", best_b)
print("matrix:\n", matrix)
