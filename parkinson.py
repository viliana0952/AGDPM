import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_labels(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")

    label_dict = {}

    for _, row in df.iterrows():
        subject = row["participant_id"]
        group = row["group"]

        if group == "Control":
            label = 0
        else:
            label = 1  # PD-NC + PD-MCI

        label_dict[subject] = label

    return label_dict


def collect_files(dataset_path, label_dict):
    files = []
    labels = []

    for sub in os.listdir(dataset_path):
        sub_path = os.path.join(dataset_path, sub)

        if sub not in label_dict:
            continue

        nii_files = glob.glob(os.path.join(sub_path, "anat", "*.nii.gz"))

        if len(nii_files) == 0:
            continue

        files.append(nii_files[0])
        labels.append(label_dict[sub])

    return files, labels


def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def preprocess(image, target_shape=(128, 128, 128)):
    # normalize 0–1
    image = image.astype(np.float32)
    image = (image - np.mean(image)) / (np.std(image) + 1e-8)

    factors = (
        target_shape[0] / image.shape[0],
        target_shape[1] / image.shape[1],
        target_shape[2] / image.shape[2],
    )

    image = zoom(image, factors)

    return image

class BrainDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        image = load_nifti(file_path)
        image = preprocess(image)

        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()

        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()

        image = np.expand_dims(image, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
    


dataset_path = r"C:\Users\Acer\Desktop\ds005892-main"
label_dict = load_labels(os.path.join(dataset_path, "participants.tsv"))
files, labels = collect_files(dataset_path, label_dict)

train_files, test_files, train_labels, test_labels = train_test_split(
    files, labels, test_size=0.2, random_state=42
)
train_dataset = BrainDataset(train_files, train_labels)
test_dataset = BrainDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# 3dResNet
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3D, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.layer2 = BasicBlock3D(16, 32, stride=2)
        self.layer3 = nn.Sequential(
            BasicBlock3D(32, 64, stride=2),
            BasicBlock3D(64, 64)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet3D(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

def train_model(model, loader, epochs=12):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

train_model(model, train_loader, epochs=12)

def evaluate(model, loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\n===== CONFUSION MATRIX COMPONENTS =====")
    print("TP (True Positive):", tp)
    print("TN (True Negative):", tn)
    print("FP (False Positive):", fp)
    print("FN (False Negative):", fn)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall (Sensitivity):", rec)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)

evaluate(model, test_loader)


# Autoencoder3D
class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Identity()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
#new dataset
class BrainDatasetAE(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = load_nifti(self.files[idx])
        image = preprocess(image)

        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32)

        return image
    
ae_dataset = BrainDatasetAE(files)
ae_loader = DataLoader(ae_dataset, batch_size=2, shuffle=True)

model_ae = Autoencoder3D().to(device)
criterion_ae = nn.MSELoss()
optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=0.0001)

def train_autoencoder(model, loader, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images in loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion_ae(outputs, images)

            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

train_autoencoder(model_ae, ae_loader, epochs=10)