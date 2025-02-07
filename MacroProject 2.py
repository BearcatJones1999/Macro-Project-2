# Parameters
s = 0.3         # Savings rate
A_t = 1.2       # Productivity
alpha = 0.33    # Capital elasticity of output
delta = 0.05    # Depreciation rate
tolerance = 1e-6  # Convergence threshold

# Initial capital
k_t = 1.0  

# Iterate until convergence
while True:
    k_t1 = s * A_t * (k_t ** alpha) + (1 - delta) * k_t  # Compute k_{t+1}
    
    if abs(k_t1 - k_t) < tolerance:  # Check for convergence
        break
    
    k_t = k_t1  # Update k_t

print(k_t)

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.33    # Capital share
s = 0.3         # Savings rate
delta = 0.05    # Depreciation rate
A_t = 1.2       # Productivity
L_t = 100       # Labor
T = 50          # Time periods

# Initial capital
k_t = 10  
capital_path = [k_t]
output_path = []

# Loop over time periods
for t in range(T):
    Y_t = np.exp(alpha * np.log(k_t) + (1 - alpha) * np.log(L_t) + np.log(A_t))
    output_path.append(Y_t)
    
    k_t1 = s * A_t * (k_t ** alpha) + (1 - delta) * k_t  
    capital_path.append(k_t1)
    
    k_t = k_t1

# 🔹 Fix: Ensure same length for plotting
time_steps = range(T)  # For T values in output_path
capital_time_steps = range(T + 1)  # For T+1 values in capital_path

# Plot the trajectories
plt.figure(figsize=(10, 5))
plt.plot(time_steps, output_path, label="Output (Y_t)", color="blue")
plt.plot(capital_time_steps, capital_path, label="Capital (K_t)", color="red", linestyle="dashed")

plt.xlabel("Time (t)")
plt.ylabel("Value")
plt.title("Trajectory of Capital and Output Over Time")
plt.legend()
plt.grid()
plt.show()

# Parameters
alpha = 0.33    # Capital share
s = 0.3         # Savings rate
delta = 0.05    # Depreciation rate
A_t = 1.2       # Productivity
L_t = 100       # Labor (assumed constant)
T = 50          # Time periods

# Calculate steady-state capital and output
K_t_steady = (s * A_t / delta) ** (1 / (1 - alpha))
Y_t_steady = A_t * (K_t_steady ** alpha)

# Initial capital
k_t = 10  
capital_path = [k_t]
output_path = []
output_per_capita_path = []

# Loop over time periods
for t in range(T):
    # Compute output based on given equation
    Y_t = np.exp(alpha * np.log(k_t) + (1 - alpha) * np.log(L_t) + np.log(A_t))
    output_path.append(Y_t)

    # Compute output per capita
    y_t = Y_t / L_t
    output_per_capita_path.append(y_t)
    
    # Update capital for next period
    k_t1 = s * A_t * (k_t ** alpha) + (1 - delta) * k_t  
    capital_path.append(k_t1)
    
    k_t = k_t1

    # Print results for each year
    print(f"Year {t + 1}: Capital (K_t) = {k_t:.2f}, Output (Y_t) = {Y_t:.2f}, Output per Capita (y_t) = {y_t:.2f}")

# Fix time step sizes for plotting
time_steps = range(T)       # Output per capita has T values
capital_time_steps = range(T + 1)  # Capital path has T+1 values

# Plot the trajectories
plt.figure(figsize=(12, 6))

# Plot Output per Capita
plt.plot(time_steps, output_per_capita_path, label="Output per Capita (y_t)", color="green")

# Plot Capital per Capita (K_t / L_t) as reference
plt.plot(capital_time_steps[:-1], np.array(capital_path[:-1]) / L_t, label="Capital per Capita (k_t)", color="red", linestyle="dashed")

# Plot Steady-State Output per Capita (constant over time)
steady_output_per_capita = Y_t_steady / L_t
plt.axhline(steady_output_per_capita, color="orange", linestyle="--", label="Steady-State Output per Capita")

plt.xlabel("Time (t)")
plt.ylabel("Value")
plt.title("Trajectory of Output per Capita, Capital per Capita and Steady-State Output per Capita")
plt.legend()
plt.grid()
plt.show()

