import numpy as np
import pandas as pd

# Fix random seed for reproducibility

np.random.seed(2025)

# Number of rows

n = 500

# Simulated sensor inputs (ranges are agronomically plausible)

soil_N = np.random.uniform(5, 80, n)      # mg/kg
soil_P = np.random.uniform(3, 15, n)      # mg/kg
soil_K = np.random.uniform(20, 160, n)    # mg/kg
pH = np.random.uniform(4.5, 8.5, n)
moisture = np.random.uniform(10, 95, n)   # %
temp = np.random.uniform(18, 36, n)       # °C
humidity = np.random.uniform(35, 98, n)   # %

# BRRI-like target requirements (kg/ha)

target_N, target_P, target_K = 92.0, 23.0, 75.0

# Soil-test to indigenous supply conversion factors (simplified STCR-inspired)

alpha_N, alpha_P, alpha_K = 0.9, 1.0, 0.7

def pH_adj(p):
return np.where(p < 5.5, 0.10 * target_N, 0.0)

def moist_adj(m):
return np.where(m > 85, -0.10 * target_N, 0.0)

# Indigenous supply

ind_N = alpha_N * soil_N
ind_P = alpha_P * soil_P
ind_K = alpha_K * soil_K

# Fertilizer requirement (kg/ha)

req_N = target_N - ind_N + pH_adj(pH) + moist_adj(moisture)
req_P = target_P - ind_P + np.where(pH < 5.5, 0.05 * target_P, 0.0)
req_K = target_K - ind_K

# Add noise and clip at 0

req_N = np.clip(req_N + np.random.normal(0, 3, n), 0, None)
req_P = np.clip(req_P + np.random.normal(0, 1.5, n), 0, None)
req_K = np.clip(req_K + np.random.normal(0, 4, n), 0, None)

# Convert to fertilizer products (kg/ha)

urea = req_N / 0.46          # Urea = 46% N
tsp = req_P / 0.46           # TSP = 46% P2O5
mop = req_K / 0.60           # MoP = 60% K2O

# Build minimal dataset

df = pd.DataFrame({
"Soil_N_mgkg": np.round(soil_N,2),
"Soil_P_mgkg": np.round(soil_P,2),
"Soil_K_mgkg": np.round(soil_K,2),
"pH": np.round(pH,2),
"Moisture_pct": np.round(moisture,2),
"Temp_C": np.round(temp,2),
"Humidity_pct": np.round(humidity,2),
"Urea_kg_ha": np.round(urea,2),
"TSP_kg_ha": np.round(tsp,2),
"MoP_kg_ha": np.round(mop,2)
})

# Save to CSV

df.to_csv("minimal_boro_fertilizer_dataset_500.csv", index=False)

print("Dataset saved as minimal_boro_fertilizer_dataset_500.csv")
print(df.head(10))
