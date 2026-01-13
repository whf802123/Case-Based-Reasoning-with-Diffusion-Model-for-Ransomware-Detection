import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# USTC-Synthetic:    E:\diffusionmodel\.venv\USTC_Malware_images_synthetic      E:\diffusionmodel\.venv\USTC_Benign_images_synthetic
# ISOT-Synthetic:    E:\diffusionmodel\.venv\ISOT_Malware_images_synthetic      E:\diffusionmodel\.venv\ISOT_Benign_images_synthetic
malware_dir = r"D:\Data\isot_Malware_images"     # USTC-Original:   D:\Data\Malware_images_16         D:\Data\Benign_images_16
benign_dir  = r"D:\Data\isot_Benign_images"       # ISOT-Original:   D:\Data\isot_Malware_images       D:\Data\isot_Benign_images

malware_files = glob.glob(os.path.join(malware_dir, "*.png"))
benign_files = glob.glob(os.path.join(benign_dir, "*.png"))
data = [(fp, 1) for fp in malware_files] + [(fp, 0) for fp in benign_files]

train_data, test_data = train_test_split(
    data,
    test_size=0.3,
    random_state=42,
    stratify=[label for (_, label) in data]
)

class TrafficDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    total_samples = len(data)
    train_samples = len(train_data)
    test_samples = len(test_data)
    print(f"Total samples: {total_samples}, Training: {train_samples}, Testing: {test_samples}")

    train_dataset = TrafficDataset(train_data, transform=transform_pipeline)
    test_dataset = TrafficDataset(test_data, transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SwinWithCrossAttention(nn.Module):
        def __init__(self):
            super(SwinWithCrossAttention, self).__init__()
            self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
            if hasattr(self.backbone, "global_pool"):
                self.backbone.global_pool = nn.Identity()
            self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, 768))

        def forward(self, x):
            tokens = self.backbone.forward_features(x)
            if tokens.dim() == 4:
                if tokens.shape[-1] == 768:
                    tokens = tokens.flatten(1, 2)
                else:
                    tokens = tokens.flatten(2).transpose(1, 2)
            B = tokens.shape[0]
            query = self.query.expand(B, -1, -1)
            attn_output, _ = self.cross_attn(query, tokens, tokens)
            attn_output = attn_output.squeeze(1)
            return attn_output


    model = SwinWithCrossAttention().to(DEVICE)
    model.eval()


    train_features = []
    train_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(DEVICE)
            feats = model(images)
            train_features.append(feats.cpu().numpy())
            train_labels.append(labels.cpu().numpy())
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    print("Train features shape:", train_features.shape)

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(train_features, train_labels)

    test_features = []
    test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            feats = model(images)
            test_features.append(feats.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    print("Test features shape:", test_features.shape)

    predictions = knn.predict(test_features)

    test_acc = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)

    print("\nFinal Evaluation on Test Set:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    probs = knn.predict_proba(test_features)[:, 1]
    fpr, tpr, thresholds = roc_curve(test_labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.show()
