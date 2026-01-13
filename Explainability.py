import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)

malware_dir = r"D:\Data\isot_Malware_images"
benign_dir  = r"D:\Data\isot_Benign_images"
image_size = (64, 64)

malware_files = glob.glob(os.path.join(malware_dir, "*.png"))
benign_files  = glob.glob(os.path.join(benign_dir, "*.png"))
data = [(fp, 1) for fp in malware_files] + [(fp, 0) for fp in benign_files]

train_data, test_data = train_test_split(
    data,
    test_size=0.3,
    stratify=[label for _, label in data],
    random_state=42
)

def extract_rgb_features(data_list):
    features_R, features_G, features_B, labels, filenames = [], [], [], [], []
    for path, label in data_list:
        image = Image.open(path).convert("RGB").resize(image_size)
        arr = np.array(image)
        R = arr[:, :, 0].flatten()
        G = arr[:, :, 1].flatten()
        B = arr[:, :, 2].flatten()
        features_R.append(R)
        features_G.append(G)
        features_B.append(B)
        labels.append(label)
        filenames.append(os.path.basename(path))
    return (
        np.stack(features_R),
        np.stack(features_G),
        np.stack(features_B),
        np.array(labels),
        filenames
    )

R_train, G_train, B_train, y_train, fnames_train = extract_rgb_features(train_data)
R_test, G_test, B_test, y_test, fnames_test = extract_rgb_features(test_data)

X_train = np.concatenate([R_train, G_train, B_train], axis=1)
X_test  = np.concatenate([R_test, G_test, B_test], axis=1)

knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_probs = knn.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve (RGB Flatten Features)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n=== RGB Channel Similarity (Top-1) on First 10 Test Samples ===")
results = []
for i in range(min(10, len(R_test))):
    R_sim = cosine_similarity(R_test[i].reshape(1, -1), R_train).max()
    G_sim = cosine_similarity(G_test[i].reshape(1, -1), G_train).max()
    B_sim = cosine_similarity(B_test[i].reshape(1, -1), B_train).max()
    results.append({
        "Filename": fnames_test[i],
        "True Label": "Ransomware" if y_test[i] == 1 else "Benign",
        "Predicted": "Ransomware" if y_pred[i] == 1 else "Benign",
        "Predicted Prob": round(y_probs[i], 4),
        "R Similarity": round(R_sim, 4),
        "G Similarity": round(G_sim, 4),
        "B Similarity": round(B_sim, 4)
    })

results_df = pd.DataFrame(results)

print(results_df)
