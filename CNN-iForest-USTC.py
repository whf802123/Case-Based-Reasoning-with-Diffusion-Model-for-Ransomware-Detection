import os
import glob
import numpy as np
from scapy.all import rdpcap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

'''
MALWARE_DIR = r"E://1/data/USTC-TFC2016/Malware"   # E://1/data/isot_app_and_botnet_dataset/botnet_data      E://1/data/USTC-TFC2016/Malware
BENIGN_DIR = r"E://1/data/USTC-TFC2016/Benign"     # E://1/data/isot_app_and_botnet_dataset/application_data      E://1/data/USTC-TFC2016/Benign
NUM_FEATURES = 2

def extract_packets_from_pcap(file_path):
    try:
        packets = rdpcap(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return []
    samples = []
    prev_time = None
    for pkt in packets:
        pkt_len = len(pkt)
        if prev_time is None:
            inter_time = 0.0
        else:
            inter_time = pkt.time - prev_time
        samples.append([pkt_len, inter_time])
        prev_time = pkt.time
    return samples

def load_packets_from_directory(directory, label):
    file_list = glob.glob(os.path.join(directory, "**", "*.pcap"), recursive=True)
    data_list = []
    labels = []
    for file in file_list:
        packets = extract_packets_from_pcap(file)
        data_list.extend(packets)
        labels.extend([label] * len(packets))
    return data_list, labels

malware_data, malware_labels = load_packets_from_directory(MALWARE_DIR, 1)
benign_data, benign_labels = load_packets_from_directory(BENIGN_DIR, 0)
all_data = malware_data + benign_data
all_labels = malware_labels + benign_labels

print(f"Total samples: {len(all_data)}")
print(f"Shape of each sample: ({NUM_FEATURES},)")

X = np.array(all_data, dtype=np.float32)
y = np.array(all_labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Shape after PCA: {X_pca.shape}")

n_samples = X_pca.shape[0]
X_reshaped = X_pca.reshape(n_samples, 2, 1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

input_shape = X_train.shape[1:]
inputs = Input(shape=input_shape)
x = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
feature_output = Dense(64, activation='relu', name='feature_output')(x)
classification_output = Dense(1, activation='sigmoid')(feature_output)

cnn_model = Model(inputs=inputs, outputs=classification_output)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

cnn_model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.1)

feature_extractor = Model(inputs=inputs, outputs=cnn_model.get_layer('feature_output').output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

isoforest = IsolationForest(contamination=0.1, random_state=42)
isoforest.fit(X_train_features)
y_pred_if = isoforest.predict(X_test_features)
y_pred = np.where(y_pred_if == 1, 0, 1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)

print("Isolation Forest Detection Results:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {roc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

'''
import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

malware_dir = r"D://Data/Malware_images_16/"
benign_dir = r"D://Data/Benign_images_16/"

malware_files = glob.glob(os.path.join(malware_dir, "*.png"))
benign_files = glob.glob(os.path.join(benign_dir, "*.png"))
data = [(fp, 1) for fp in malware_files] + [(fp, 0) for fp in benign_files]
np.random.shuffle(data)
file_paths, labels = zip(*data)
labels = np.array(labels)

img_height, img_width = 64, 64

def load_and_preprocess_image(fp):
    img = Image.open(fp).convert("L")
    img = img.resize((img_width, img_height))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

X = np.array([load_and_preprocess_image(fp) for fp in file_paths])
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

input_shape = X_train.shape[1:]
inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
feature_output = Dense(64, activation='relu', name='feature_output')(x)
classification_output = Dense(1, activation='sigmoid')(feature_output)

cnn_model = Model(inputs=inputs, outputs=classification_output)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stop])

feature_extractor = Model(inputs=inputs, outputs=cnn_model.get_layer('feature_output').output)
X_train_features = feature_extractor.predict(X_train, batch_size=32)
X_test_features = feature_extractor.predict(X_test, batch_size=32)

isoforest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
isoforest.fit(X_train_features)
y_pred_if = isoforest.predict(X_test_features)
y_pred = np.where(y_pred_if == 1, 0, 1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)

print("Isolation Forest Detection Results:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {roc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
