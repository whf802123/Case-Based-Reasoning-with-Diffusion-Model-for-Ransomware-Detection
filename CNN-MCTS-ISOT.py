import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

malware_dir = r"D:\Data\isot_Malware_images"
benign_dir = r"D:\Data\isot_Benign_images"

malware_files = glob.glob(os.path.join(malware_dir, "*.png"))
benign_files = glob.glob(os.path.join(benign_dir, "*.png"))

data = [(fp, 1) for fp in malware_files] + [(fp, 0) for fp in benign_files]

def load_data(data, img_size=(224, 224)):
    images = []
    labels = []
    for fp, label in data:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        images.append(img)
        labels.append(label)
    images = np.array(images, dtype='float32') / 255.0  # 归一化
    labels = np.array(labels)
    return images, labels

images, labels = load_data(data)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn((224, 224, 3))
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)

cnn_preds = model.predict(X_test)

def mcts_decision_optimization(features, cnn_prob, num_iterations=10):
    optimized_prob = cnn_prob
    for i in range(num_iterations):
        adjustment = np.random.uniform(-0.01, 0.01)
        optimized_prob += adjustment
        optimized_prob = np.clip(optimized_prob, 0, 1)
    return optimized_prob

optimized_preds = np.array([mcts_decision_optimization(None, p) for p in cnn_preds])

binary_preds = (optimized_preds >= 0.5).astype(int)

accuracy = accuracy_score(y_test, binary_preds)
cm = confusion_matrix(y_test, binary_preds)
precision = precision_score(y_test, binary_preds)
recall = recall_score(y_test, binary_preds)
f1 = f1_score(y_test, binary_preds)

fpr, tpr, _ = roc_curve(y_test, optimized_preds)
roc_auc = auc(fpr, tpr)

print("Test Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 参考线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
