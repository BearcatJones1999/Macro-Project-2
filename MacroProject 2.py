import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file (adjust file path as necessary)
file_path = "C:/Users/koush/OneDrive - University of Cincinnati/R/Macro/Penn Data Proj 2.xlsx"

# Load the 4th sheet (index 3, since Python uses zero-based indexing)
sheet_name = 3
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Select the relevant columns
df_filtered = df[['countrycode', 'year', 'rgdpo', 'rnna', 'rtfpna', 'labsh', 'emp']]

# Filter for the selected country 
country_code = "USA"
df_country = df_filtered[df_filtered['countrycode'] == country_code].copy()

# Drop rows with missing values in key columns
df_country.dropna(subset=['rgdpo', 'rnna', 'emp'], inplace=True)

# Compute log values
df_country['log_Y'] = np.log(df_country['rgdpo'])
df_country['log_K'] = np.log(df_country['rnna'])
df_country['log_L'] = np.log(df_country['emp'])

# Compute log(A_t) using the Solow residual formula
alpha = 0.3
df_country['log_A'] = df_country['log_Y'] - (alpha * df_country['log_K']) - ((1 - alpha) * df_country['log_L'])

# Normalize A_t to 1 in the base year (1980)
base_year = 1980
A_base = df_country[df_country['year'] == base_year]['log_A'].values[0]
df_country['A_t'] = np.exp(df_country['log_A'] - A_base)  # Convert back from log scale

# Plot A_t over time
plt.figure(figsize=(10, 5))
plt.plot(df_country['year'], df_country['A_t'], marker='o', linestyle='-', label="Solow Residual (A_t)", color="orange")
plt.axhline(y=1, color='r', linestyle='--', label="Base Year (1980)")
plt.xlabel("Year")
plt.ylabel("Solow Residual (A_t)")
plt.title("Solow Residual Over Time (USA)")
plt.legend()
plt.grid

# Parameters
s = 0.038         # Savings rate
alpha = 0.33      # Capital elasticity of output
delta = 0.037     # Depreciation rate
tolerance = 1e-6  # Convergence threshold

# Ensure data is sorted by year
df_country = df_country.sort_values(by='year').reset_index(drop=True)

# Initialize capital using the first available year
first_year_index = df_country['year'].idxmin()  # Index of the first year
k_t = (df_country.loc[first_year_index, 'rnna'] / df_country.loc[first_year_index, 'emp'])  # Initial capital per worker

k_t_values = []  # Store k_t for each year

# Iterate over the years, checking for convergence
for _, row in df_country.iterrows():
    A_t = row['A_t']
    
    # Compute k_{t+1} using the model
    k_t1 = s * A_t * (k_t ** alpha) + (1 - delta) * k_t  # Compute k_{t+1} using the model
    
    # Append current capital stock before updating
    k_t_values.append(k_t)
    
    # Check for convergence (if the change in capital is smaller than tolerance, stop)
    if abs(k_t1 - k_t) < tolerance:
        print(f"Converged to steady-state capital at year {row['year']} with k_t = {k_t1:.4f}")
        break
    
    # Update k_t for next iteration
    k_t = k_t1  

# Ensure the length of k_t_values matches df_country
df_country['k_t'] = k_t_values  

# Plot capital accumulation over time
plt.figure(figsize=(10, 5))
plt.plot(df_country['year'], df_country['k_t'], marker='o', linestyle='-', label="Capital Stock (k_t)", color="blue")
plt.xlabel("Year")
plt.ylabel("Capital Stock (k_t)")
plt.title("Capital Stock Over Time (USA)")
plt.legend()
plt.grid(True)
plt.show()



# Parameters
s = 0.038         # Savings rate
alpha = 0.33     # Capital elasticity of output
delta = 0.037    # Depreciation rate

# Get time periods from df_country
T = len(df_country)  

# Initialize output list
output_path = []

# Iterate over each year in df_country
for _, row in df_country.iterrows():
    A_t = row['A_t']  # Use A_t dynamically from DataFrame
    L_t = row['emp']  # Use L_t dynamically from DataFrame
    k_t = row['k_t']  # Use k_t directly from DataFrame (already in df_country)

    # Compute output Y_t
    Y_t = np.exp(alpha * np.log(k_t) + (1 - alpha) * np.log(L_t) + np.log(A_t))
    output_path.append(Y_t)

# 🔹 Fix: Ensure same length for plotting
time_steps = df_country['year']  # Use actual years from DataFrame

# Plot Output (Y_t)
plt.figure(figsize=(12, 6))
plt.plot(time_steps, output_path, label="Output (Y_t)", color="blue")
plt.plot(df_country['year'], df_country['rgdpo'], label="Actual Output (rgdpo)", color="green", linestyle="--")
plt.xlabel("Time (t)")
plt.ylabel("Output (Y_t)")
plt.title("Trajectory of Output Over Time")
plt.legend()
plt.grid(True)
plt.show()


# Part 3

# Parameters
alpha = 0.33    # Capital share
s = 0.38        # Savings rate
delta = 0.037   # Depreciation rate

# Ensure df_country contains A_t and L_t over time
A_t_series = df_country['A_t'].values  # Extract A_t as an array
L_t_series = df_country['emp'].values  # Extract labor force (L_t) as an array

# Ensure A_t_series and L_t_series are long enough for T periods
T = len(df_country)
assert len(A_t_series) >= T, "Error: A_t series is shorter than simulation time."
assert len(L_t_series) >= T, "Error: L_t series is shorter than simulation time."

# Calculate steady-state capital and output using the first A_t
K_t_steady = (s * A_t_series[0] / delta) ** (1 / (1 - alpha))
Y_t_steady = A_t_series[0] * (K_t_steady ** alpha)

# Initial capital
k_t = 10  
capital_path = [k_t]
output_path = []
output_per_capita_path = []

# Loop over time periods
for t in range(T):
    A_t = A_t_series[t]  # Dynamically reference A_t over time
    L_t = L_t_series[t]  # Dynamically reference L_t over time

    # Compute output based on evolving A_t
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
    print(f"Year {t + 1}: A_t = {A_t:.2f}, Capital (K_t) = {k_t:.2f}, Output (Y_t) = {Y_t:.2f}, Output per Capita (y_t) = {y_t:.2f}")

# Fix time step sizes for plotting
time_steps = df_country['year']  # Use actual years from DataFrame
capital_time_steps = list(df_country['year']) + [df_country['year'].iloc[-1] + 1]  # Extend by 1 for k_t

# Plot Output per Capita vs. Steady-State Output per Capita
plt.figure(figsize=(12, 6))

# Plot Output per Capita
plt.plot(time_steps, output_per_capita_path, label="Output per Capita (y_t)", color="green")

plt.xlabel("Time (t)")
plt.ylabel("Output per Capita (y_t)")
plt.title("Trajectory of Output per Capita")
plt.legend()
plt.grid()

plt.show()

# Plot Data Output vs Steady-State Output (Comparing model with data)
plt.figure(figsize=(12, 6))

# Plot Model Output (Y_t) vs Data Output (rgdpo)
plt.plot(time_steps, output_path, label="Model Output (Y_t)", color="blue")
plt.plot(time_steps, df_country['rgdpo'], label="Data Output (rgdpo)", color="red", linestyle="--")

plt.xlabel("Time (t)")
plt.ylabel("Output (Y_t / rgdpo)")
plt.title("Comparison of Model Output and Data Output Over Time")
plt.legend()
plt.grid()

plt.show()



