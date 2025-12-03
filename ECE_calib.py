import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Load dataset
df = pd.read_csv("D:\Human_AI\human-aligned-calibration\human_ai_interactions_data\haiid_dataset.csv")
#
# Filter: only US & perceived accuracy = 80
# df = df[(df['geographic_region'] == 'United States') & (df['perceived_accuracy'] == 80)]
# Keep only relevant columns
df = df[['task_instance_id','participant_id','correct_label','advice','response_1','response_2']]

# --- Assign y (binary event) ---
# Random assignment of which label = 1 for each task_instance
np.random.seed(320)
task_ids = df['task_instance_id'].unique()
df_label = pd.DataFrame(task_ids, columns=['task_instance_id'])
df_label['y'] = np.random.choice(2, len(task_ids)).astype(int)
df = df.merge(df_label, on='task_instance_id')

# --- Map [-1,1] → [0,1] for advice, human, human+AI ---
df[['b','h','h+AI']] = (df[['advice','response_1','response_2']] + 1) / 2.0
df.loc[df['y']==0, ['b','h','h+AI']] = 1 - df.loc[df['y']==0, ['b','h','h+AI']]

# --- Helper: compute calibration + ECE ---
def compute_calibration(y_true, y_pred, label, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

    # Expected Calibration Error (ECE)
    bin_sizes = np.histogram(y_pred, bins=n_bins)[0]
    bin_weights = bin_sizes / bin_sizes.sum()
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))

    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f"{label} (ECE={ece:.3f})")
    return ece

# --- Plot all calibration curves together ---
plt.figure(figsize=(6,6))
ece_b   = compute_calibration(df['y'], df['b'],   "Advice (b)")
ece_h   = compute_calibration(df['y'], df['h'],   "Human (h)")
ece_hAI = compute_calibration(df['y'], df['h+AI'], "Human+AI (h+AI)")

plt.plot([0,1],[0,1], linestyle="--", color="gray", label="Perfect Calibration")
plt.xlabel("Predicted Probability", fontsize=14)
plt.ylabel("Empirical Probability", fontsize=14)
plt.legend(fontsize=12)
plt.title("Calibration Curves Across All Tasks", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

print(f"ECE (Advice):   {ece_b:.4f}")
print(f"ECE (Human):    {ece_h:.4f}")
print(f"ECE (Human+AI): {ece_hAI:.4f}")