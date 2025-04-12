import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from fpdf import FPDF

# Load dataset
file_path = "/storage/emulated/0/Download/S11 port1.txt"
data = pd.read_csv(file_path, delimiter="\t", comment='#', header=None)

# Rename columns
data.columns = ['Frequency', 'S11']

# Splitting dataset
X = data[['Frequency']]
y = data['S11']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM Regressor": SVR(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
}

# Store results
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    r2 = r2_score(y_test, y_pred)
    var_score = explained_variance_score(y_test, y_pred)

    results.append([name, mae, mse, rmse, mape, r2, var_score])

# Convert to DataFrame
metrics_df = pd.DataFrame(results, columns=["Classifier", "MAE", "MSE", "RMSE", "MAPE", "R²", "Variance Score"])

# Save results to CSV
csv_path = "/storage/emulated/0/Download/classifier_results.csv"
metrics_df.to_csv(csv_path, index=False)

# Identify Best Classifier
best_classifier = metrics_df.sort_values(by=["MAE", "MSE", "RMSE", "MAPE"], ascending=True).iloc[0]

print(f"Results saved as CSV: {csv_path}")


#  Load S11 Data (Replace with your file path)
#file_path = "/storage/emulated/0/Download/S11 port1.txt"
data = pd.read_csv(file_path, delimiter="\t", skiprows=3, names=["Frequency_THz", "S11_dB"])

# Convert Frequency from THz to GHz
data["Frequency_GHz"] = data["Frequency_THz"] * 1000  
freq_GHz = data["Frequency_GHz"].values.reshape(-1, 1)
S11_dB = data["S11_dB"].values

# Compute All 11 Parameters
S11_linear = 10 ** (S11_dB / 20)
VSWR = (1 + np.abs(S11_linear)) / (1 - np.abs(S11_linear))
radiation_efficiency = (1 - np.abs(S11_linear) ** 2) * 100
group_delay = -np.gradient(S11_dB, data["Frequency_GHz"])
avg_group_delay = np.mean(group_delay) * 1e9
Z0 = 50
Zin = Z0 * (1 + S11_linear) / (1 - S11_linear)

# **Far-Field Radiation Pattern Calculation**
theta = np.linspace(0, 2 * np.pi, 360)  # 360 points for smooth pattern
radiation_pattern = np.abs(np.cos(theta))  # Example pattern

#  **Bandwidth Calculation**
# Assume bandwidth is the frequency range where |S11| <= -10 dB
min_s11_threshold = -10
bandwidth_range = data[data["S11_dB"] <= min_s11_threshold]
bandwidth = bandwidth_range["Frequency_GHz"].iloc[-1] - bandwidth_range["Frequency_GHz"].iloc[0]

# Define Parameters for Classifier Comparison
parameters = {
    "Return Loss (S11)": S11_dB,
    "VSWR": VSWR,
    "Radiation Efficiency": radiation_efficiency,
    "Group Delay": group_delay * 1e9,
    "Impedance Matching (Real)": Zin.real,
    "Impedance Matching (Imag)": Zin.imag,
    "Bandwidth": [bandwidth] * len(freq_GHz),  # Bandwidth is a single value for all frequencies
    "Group Delay Average": [avg_group_delay] * len(freq_GHz)  # Group Delay Average is constant
}

#  Machine Learning-Based Optimization for All Parameters
classifiers = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM Regressor": SVR(kernel="rbf"),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
}

results = {param: {} for param in parameters}
for param, values in parameters.items():
    values = np.array(values).reshape(-1, 1)  
    for name, model in classifiers.items():
        model.fit(freq_GHz, values.ravel())  
        predicted_values = model.predict(freq_GHz)  
        mse = mean_squared_error(values, predicted_values)
        r2 = r2_score(values, predicted_values)
        results[param][name] = {"MSE": mse, "R2 Score": r2, "Predicted": predicted_values}

# Find the Best Classifier for Each Parameter
best_classifiers = {param: min(results[param], key=lambda x: results[param][x]["MSE"]) for param in parameters}

# **Return Loss Optimization**: Find the frequency where return loss (S11) is minimized
return_loss_min_idx = np.argmin(S11_dB)
return_loss_min_freq = data["Frequency_GHz"].values[return_loss_min_idx]
return_loss_min_value = S11_dB[return_loss_min_idx]

# **Resonant Frequency Identification**: Identifying the resonant frequency at minimum S11
resonant_frequency = return_loss_min_freq  # Resonant frequency is the same as min S11 frequency

# **Polarization Analysis**: Simple polarization angle analysis based on S11
# Polarization angle (using a simple model for visualization purposes)
polarization_angle = np.degrees(np.angle(S11_linear))

#Plot Formatting
label_size, title_size, tick_size, legend_size = 14, 20, 14, 14

#  Generate and Save Plots for All Parameters
for param, values in parameters.items():
    plt.figure(figsize=(8, 5))
    plt.plot(freq_GHz, values, label=f"Actual {param}", color="blue", linewidth=2)
    for name, res in results[param].items():
        plt.plot(freq_GHz, res["Predicted"], linestyle="dashed", label=f"{name} (MSE: {res['MSE']:.4f})")
    plt.xlabel("Frequency (GHz)", fontsize=label_size, fontweight='bold')
    plt.ylabel(param, fontsize=label_size, fontweight='bold')
    plt.title(f"{param} Optimization (ML Classifiers)", fontsize=title_size, fontweight='bold', pad=15)

    # Adjust legend position for **Impedance Matching**
    if "Impedance Matching" in param:
        plt.legend(fontsize=legend_size - 2, loc="upper right", bbox_to_anchor=(1.1, 1.0))
    else:
        plt.legend(fontsize=legend_size - 2)

    plt.xticks(fontsize=tick_size, fontweight='bold')
    plt.yticks(fontsize=tick_size, fontweight='bold')
    plt.grid(True, linewidth=1.2)
    plt.tight_layout()
    plt.savefig(f"{param.replace(' ', '_')}_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

# **Return Loss Optimization Plot with Resonant Frequency**
plt.figure(figsize=(8, 5))
plt.plot(data["Frequency_GHz"], S11_dB, label="Return Loss (S11)", color="blue", linewidth=2)

# Mark the Minimum S11 Point
plt.scatter(return_loss_min_freq, return_loss_min_value, color="red", zorder=5, label=f"Min S11: {return_loss_min_value:.2f} dB at {return_loss_min_freq:.2f} GHz")

# Mark the Resonant Frequency
plt.axvline(x=resonant_frequency, color="green", linestyle="--", label=f"Resonant Frequency: {resonant_frequency:.2f} GHz")

plt.xlabel("Frequency (GHz)", fontsize=label_size, fontweight='bold')
plt.ylabel("S11 (dB)", fontsize=label_size, fontweight='bold')
plt.title("Return Loss Optimization with Resonant Frequency", fontsize=title_size, fontweight='bold', pad=15)
plt.legend(fontsize=legend_size)
plt.grid(True, linewidth=1.2)
plt.tight_layout()

#  Save the Return Loss Plot with Resonant Frequency
plt.savefig("return_loss_optimization_with_resonant_frequency.png", dpi=300, bbox_inches="tight")
plt.close()

#  **Far-Field Radiation Pattern Plot**
plt.figure(figsize=(6, 6))
plt.polar(theta, radiation_pattern, label="Radiation Pattern")
plt.title("Far-Field Radiation Pattern", fontsize=title_size, fontweight='bold', pad=20)
plt.legend(fontsize=legend_size, loc="upper right")
plt.tight_layout()
plt.savefig("far_field_radiation_pattern.png", dpi=300, bbox_inches="tight")
plt.close()

#  **Polarization Analysis Plot**
plt.figure(figsize=(8, 5))
plt.plot(freq_GHz, polarization_angle, color="green", linewidth=2)
plt.xlabel("Frequency (GHz)", fontsize=label_size, fontweight='bold')
plt.ylabel("Polarization Angle (Degrees)", fontsize=label_size, fontweight='bold')
plt.title("Polarization Angle vs Frequency", fontsize=title_size, fontweight='bold', pad=15)
plt.grid(True, linewidth=1.2)
plt.tight_layout()
plt.savefig("polarization_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# **Generate PDF Report**
# Save results to PDF with proper formatting
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Classifier Performance Report", ln=True, align='C')
pdf.ln(10)

# Best Classifier Summary
pdf.set_font("Arial", "B", 12)
pdf.cell(200, 10, "Best Classifier:", ln=True, align='L')
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, f"The best classifier is **{best_classifier['Classifier']}**, as it has the lowest "
                      f"MAE ({best_classifier['MAE']:.4f}), MSE ({best_classifier['MSE']:.4f}), "
                      f"RMSE ({best_classifier['RMSE']:.4f}), and MAPE ({best_classifier['MAPE']:.4f}). "
                      "This indicates higher accuracy and better performance.", align='L')
pdf.ln(10)

# Add Table Header
pdf.set_font("Arial", "B", 10)
pdf.cell(40, 10, "Classifier", 1, 0, "C")
pdf.cell(20, 10, "MAE", 1, 0, "C")
pdf.cell(20, 10, "MSE", 1, 0, "C")
pdf.cell(20, 10, "RMSE", 1, 0, "C")
pdf.cell(20, 10, "MAPE", 1, 0, "C")
pdf.cell(20, 10, "R²", 1, 0, "C")
pdf.cell(30, 10, "Variance Score", 1, 1, "C")

# Add Table Rows
pdf.set_font("Arial", "", 10)
for index, row in metrics_df.iterrows():
    pdf.cell(40, 10, row["Classifier"], 1, 0, "C")
    pdf.cell(20, 10, f"{row['MAE']:.4f}", 1, 0, "C")
    pdf.cell(20, 10, f"{row['MSE']:.4f}", 1, 0, "C")
    pdf.cell(20, 10, f"{row['RMSE']:.4f}", 1, 0, "C")
    pdf.cell(20, 10, f"{row['MAPE']:.4f}", 1, 0, "C")
    pdf.cell(20, 10, f"{row['R²']:.4f}", 1, 0, "C")
    pdf.cell(30, 10, f"{row['Variance Score']:.4f}", 1, 1, "C")

# Save PDF
#pdf_path = "/storage/emulated/0/Download/classifier_results.pdf"
#pdf.output(pdf_path)

#  Summary Text
summary_text = "Single-Port MIMO Antenna Report\n\n"
summary_text += "Classifier Comparison Results:\n"
for param in best_classifiers:
    summary_text += f"  - {param}: Best Classifier = {best_classifiers[param]}, MSE = {results[param][best_classifiers[param]]['MSE']:.4f}, R² Score = {results[param][best_classifiers[param]]['R2 Score']:.4f}\n"

summary_text += "\nReturn Loss Optimization:\n"
summary_text += f"  - Frequency with Minimum Return Loss: {return_loss_min_freq:.2f} GHz, S11 (dB): {return_loss_min_value:.2f} dB\n"
summary_text += f"  - Resonant Frequency: {resonant_frequency:.2f} GHz (Minimum S11 Frequency)\n"

summary_text += "\n Bandwidth Calculation:\n"
summary_text += f"  - Bandwidth (where |S11| <= -10 dB): {bandwidth:.2f} GHz\n"

summary_text += "\n Group Delay Average:\n"
summary_text += f"  - Average Group Delay: {avg_group_delay:.2f} ns\n"

summary_text += "\nPolarization Analysis:\n"
summary_text += "  - Polarization angles are plotted versus frequency."

summary_text += "\n\nAll results and figures are presented in this report."
pdf.set_font("Arial", size=14)
for line in summary_text.split("\n"):
    pdf.cell(200, 8, line.encode('latin-1', 'replace').decode('latin-1'), ln=True)

# Add Comparison Plots to PDF
for param in parameters:
    pdf.add_page()
    pdf.image(f"{param.replace(' ', '_')}_comparison.png", x=10, y=20, w=180)

# Add **Far-Field Radiation Pattern** to PDF
pdf.add_page()
pdf.image("far_field_radiation_pattern.png", x=10, y=20, w=180)

# Add **Polarization Analysis** to PDF
pdf.add_page()
pdf.image("polarization_analysis.png", x=10, y=20, w=180)

# Save PDF Report
pdf_file_path = "Single_Port_MIMO_Classifier_Comparison.pdf"
pdf.output(pdf_file_path, "F")
#pdf.output(pdf_file_path)

print(f"Report Saved: {pdf_file_path}")
