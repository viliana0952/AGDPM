

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt


# ============================================================
# ИЗБОР НА МОДАЛНОСТИ: T1w + FLAIR вместо MRI + PET/SPECT
# ============================================================
#
# Инструкциите препоръчват за епилепсия fusion на структурно MRI
# с функционален образ (PET или SPECT) за локализация на seizure foci.
#
# Причина за използване на T1w + FLAIR вместо MRI + PET:
#
# 1. Липса на свободно достъпен epilepsy PET/SPECT dataset в OpenNeuro.
#    Всички публично достъпни epilepsy datasets в OpenNeuro (включително
#    ds005602 – IDEAS) съдържат само структурни MRI последователности
#    (T1w, FLAIR, T2w). PET/SPECT данни за епилепсия са клинично
#    чувствителни и не се публикуват свободно.
#
# 2. T1w + FLAIR е клинично валидна алтернатива за епилепсия.
#    FLAIR (Fluid-Attenuated Inversion Recovery) е чувствителен към
#    кортикални лезии, фокална кортикална дисплазия и хипокампална
#    склероза — типичните структурни промени при епилепсия. Комбинацията
#    T1w (анатомия) + FLAIR (патология) осигурява допълваща информация,
#    аналогична на structural + functional fusion.
#
# 3. Fusion стратегията (0.5 * T1w + 0.5 * FLAIR) следва принципа на
#    early image-level fusion, описан в курсовите материали, и запазва
#    непроменена архитектурата на невронната мрежа (Conv3d(1, ...)).
#
# ============================================================
# 1. CONFIG
# ============================================================

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds005602")
TARGET_SHAPE = (128, 128, 128)

BATCH_SIZE = 2
RESNET_EPOCHS = 12
AE_EPOCHS = 10

LEARNING_RATE_RESNET = 0.00001
LEARNING_RATE_AE = 0.0001

RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 2. LOAD LABELS
# ============================================================

def load_labels(tsv_path):
    """
    Зарежда labels от participants.tsv.

    Идеята е binary classification:
    0 = control / healthy
    1 = epilepsy / patient

    Първо търсим стандартни колони, които може да съдържат груповата информация.
    Ако намерим такава колона, използваме нея.
    Ако не намерим, правим fallback: проверяваме дали някой от редовете съдържа "control" или "healthy".
        - Ако съдържа -> label = 0
        - Иначе -> label = 1
        
    """

    df = pd.read_csv(tsv_path, sep="\t")

    print("\n===== participants.tsv columns =====")
    print(df.columns.tolist())
    print("\n===== participants.tsv preview =====")
    print(df.head())

    label_dict = {}

    possible_label_columns = [
        "group",
        "diagnosis",
        "participant_type",
        "type",
        "status",
        "patient_group"
    ]

    found_label_column = None

    for col in possible_label_columns:
        if col in df.columns:
            found_label_column = col
            break

    if found_label_column:
        print(f"\nИзползва се label колона: {found_label_column}")

        for _, row in df.iterrows():
            subject = row["participant_id"]
            value = str(row[found_label_column]).lower()

            if "control" in value or "healthy" in value or value == "hc":
                label = 0
            else:
                label = 1

            label_dict[subject] = label

    else:
        print("\nНе е намерена стандартна label колона.")
        print("Ще се използва fallback: ако редът съдържа control/healthy -> 0, иначе -> 1.")

        for _, row in df.iterrows():
            subject = row["participant_id"]
            row_text = " ".join([str(x).lower() for x in row.values])

            if "control" in row_text or "healthy" in row_text:
                label = 0
            else:
                label = 1

            label_dict[subject] = label

    return label_dict


# ============================================================
# 3. COLLECT T1w + FLAIR FILES
# ============================================================

def collect_fusion_files(dataset_path, label_dict):
    """
    Намира subjects, които имат и T1w, и FLAIR.

    Защо T1w + FLAIR?
    - T1w MRI показва добра анатомична структура.
    - FLAIR MRI е чувствителен към патологични промени и лезии.
    - При епилепсия често се търсят структурни аномалии,
      например фокална кортикална дисплазия.

    Защо ги сливаме в един обем?
    Оригиналната ResNet3D мрежа започва с:
        nn.Conv3d(1, 16, ...)

    Това означава, че тя приема 1 входен канал.
    Ако подадем T1 и FLAIR като 2 канала, ще трябва да сменим Conv3d(1, 16)
    на Conv3d(2, 16), което е промяна на невронната мрежа.

    Затова правим image-level fusion:
        fused = 0.5 * T1w + 0.5 * FLAIR

    Така:
    - имаме реално fused 3D изображение
    - запазваме същата невронна мрежа
    - входът остава (1, 128, 128, 128)
    """

    t1_files = []
    flair_files = []
    labels = []
    subjects_used = []

    subjects = sorted([x for x in os.listdir(dataset_path) if x.startswith("sub-")])

    for sub in subjects:
        if sub not in label_dict:
            continue

        anat_path = os.path.join(dataset_path, sub, "anat")

        if not os.path.exists(anat_path):
            continue

        t1_candidates = glob.glob(os.path.join(anat_path, "*T1w.nii.gz"))
        flair_candidates = glob.glob(os.path.join(anat_path, "*FLAIR.nii.gz"))

        if len(t1_candidates) == 0 or len(flair_candidates) == 0:
            continue

        t1_files.append(t1_candidates[0])
        flair_files.append(flair_candidates[0])
        labels.append(label_dict[sub])
        subjects_used.append(sub)

    print("\n===== Dataset summary =====")
    print("Subjects with T1w + FLAIR:", len(labels))
    print("Controls (0):", np.sum(np.array(labels) == 0))
    print("Epilepsy (1):", np.sum(np.array(labels) == 1))

    if len(labels) > 0:
        print("\nExample subject:", subjects_used[0])
        print("Example T1w:", t1_files[0])
        print("Example FLAIR:", flair_files[0])

    return t1_files, flair_files, labels, subjects_used


# ============================================================
# 4. NIFTI LOAD, PREPROCESSING, FUSION
# ============================================================

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data


def preprocess(image, target_shape=TARGET_SHAPE):
    """
    Preprocessing:
    1. Convert to float32
    2. Z-score normalization (z-score: μ=0, σ=1)
    3. Resize до 128x128x128 с билинейна интерполация (order=1)
    """

    image = image.astype(np.float32)

    image = (image - np.mean(image)) / (np.std(image) + 1e-8)

    factors = (
        target_shape[0] / image.shape[0],
        target_shape[1] / image.shape[1],
        target_shape[2] / image.shape[2],
    )

    image = zoom(image, factors, order=1)

    return image


def fuse_t1_flair(t1_image, flair_image):
    """
    Early image-level fusion на T1w и FLAIR.

    Зареждаме двата модалитета отделно, нормализираме ги независимо
    и ги сливаме voxel-wise преди подаване в мрежата.

    Формула:
        fused = 0.5 * T1w + 0.5 * FLAIR

    Защо T1w + FLAIR, а не MRI + PET/SPECT?
    Инструкциите препоръчват structural + functional fusion за епилепсия,
    но свободно достъпен epilepsy PET/SPECT dataset в OpenNeuro не е наличен.
    ds005602 (IDEAS) предоставя само структурни MRI. FLAIR е клинично
    установен за детекция на кортикални лезии при епилепсия и играе
    ролята на "функционален" proxy спрямо T1w анатомията.

    Защо 0.5 и 0.5?
    - Проста, обяснима и стабилна early fusion стратегия.
    - Не добавя нови параметри, не променя архитектурата.
    - Двата модалитета имат еднакво клинично значение за диагнозата.
    """

    fused = 0.5 * t1_image + 0.5 * flair_image
    return fused.astype(np.float32)


# ============================================================
# 5. DATASET FOR CLASSIFICATION
# ============================================================

class BrainFusionDataset(Dataset):
    def __init__(self, t1_files, flair_files, labels, augment=True):
        self.t1_files = t1_files
        self.flair_files = flair_files
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t1 = load_nifti(self.t1_files[idx])
        flair = load_nifti(self.flair_files[idx])

        t1 = preprocess(t1)
        flair = preprocess(flair)

        image = fuse_t1_flair(t1, flair)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()

            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()

        image = np.expand_dims(image, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


# ============================================================
# 6. RESNET3D
# ============================================================

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


# ============================================================
# 7. TRAIN AND EVALUATE RESNET
# ============================================================

def train_model(model, loader, criterion, optimizer, epochs=12):
    model.train()

    history_loss = []
    history_acc = []

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

        history_loss.append(total_loss)
        history_acc.append(acc)

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

    return history_loss, history_acc


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
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        print("\n===== CONFUSION MATRIX COMPONENTS =====")
        print("TP (True Positive):", tp)
        print("TN (True Negative):", tn)
        print("FP (False Positive):", fp)
        print("FN (False Negative):", fn)

    print("\nAccuracy:", acc)
    print("Precision:", prec)
    print("Recall / Sensitivity:", rec)

    return acc, prec, rec, cm


# ============================================================
# 8. AUTOENCODER3D
# ============================================================

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


class BrainFusionDatasetAE(Dataset):
    def __init__(self, t1_files, flair_files):
        self.t1_files = t1_files
        self.flair_files = flair_files

    def __len__(self):
        return len(self.t1_files)

    def __getitem__(self, idx):
        t1 = load_nifti(self.t1_files[idx])
        flair = load_nifti(self.flair_files[idx])

        t1 = preprocess(t1)
        flair = preprocess(flair)

        image = fuse_t1_flair(t1, flair)

        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32)

        return image


def train_autoencoder(model, loader, criterion, optimizer, epochs=10):
    model.train()

    history_loss = []

    for epoch in range(epochs):
        total_loss = 0

        for images in loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        history_loss.append(total_loss)

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    return history_loss


# ============================================================
# 9. PLOTS
# ============================================================

def plot_history(values, title, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def visualize_autoencoder(model_ae, ae_dataset, output_path):
    model_ae.eval()

    sample = ae_dataset[0].unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = model_ae(sample)

    sample_np = sample.cpu().numpy()[0, 0]
    recon_np = reconstructed.cpu().numpy()[0, 0]

    slice_idx = sample_np.shape[2] // 2

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(sample_np[:, :, slice_idx], cmap="gray")
    plt.title("Original fused T1w + FLAIR")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(recon_np[:, :, slice_idx], cmap="gray")
    plt.title("Reconstructed by Autoencoder3D")
    plt.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 10. MAIN
# ============================================================

def main():
    participants_path = os.path.join(DATASET_PATH, "participants.tsv")

    if not os.path.exists(participants_path):
        raise FileNotFoundError(
            f"Не е намерен participants.tsv тук: {participants_path}\n"
            f"Провери DATASET_PATH."
        )

    label_dict = load_labels(participants_path)

    t1_files, flair_files, labels, subjects_used = collect_fusion_files(DATASET_PATH, label_dict)

    if len(labels) == 0:
        raise RuntimeError(
            "Не са намерени subjects с T1w + FLAIR.\n"
            "Провери дали dataset папката съдържа sub-*/anat/*T1w.nii.gz и *FLAIR.nii.gz."
        )

    # Ако имаме само един клас, classification няма да е валиден.
    unique_labels = np.unique(labels)
    print("\nUnique labels:", unique_labels)

    if len(unique_labels) < 2:
        raise RuntimeError(
            "Има само един клас в labels.\n"
            "Трябва да коригираме load_labels(), защото binary classification изисква control и epilepsy."
        )

    train_t1, test_t1, train_flair, test_flair, train_labels, test_labels = train_test_split(
        t1_files,
        flair_files,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    train_dataset = BrainFusionDataset(train_t1, train_flair, train_labels, augment=True)
    test_dataset = BrainFusionDataset(test_t1, test_flair, test_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nTrain samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    # -----------------------------
    # ResNet3D training
    # -----------------------------

    print("\n===== Training ResNet3D =====")

    model = ResNet3D(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_RESNET)

    resnet_loss, resnet_acc = train_model(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=RESNET_EPOCHS
    )

    print("\n===== Evaluating ResNet3D =====")
    evaluate(model, test_loader)

    plot_history(
        resnet_loss,
        "ResNet3D Training Loss",
        "Loss",
        "resnet3d_training_loss.png"
    )

    plot_history(
        resnet_acc,
        "ResNet3D Training Accuracy",
        "Accuracy (%)",
        "resnet3d_training_accuracy.png"
    )

    torch.save(model.state_dict(), "resnet3d_epilepsy_fusion.pth")
    print("Saved: resnet3d_epilepsy_fusion.pth")

    # -----------------------------
    # Autoencoder3D training
    # -----------------------------

    print("\n===== Training Autoencoder3D =====")

    ae_dataset = BrainFusionDatasetAE(t1_files, flair_files)
    ae_loader = DataLoader(ae_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model_ae = Autoencoder3D().to(device)

    criterion_ae = nn.MSELoss()
    optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=LEARNING_RATE_AE)

    ae_loss = train_autoencoder(
        model=model_ae,
        loader=ae_loader,
        criterion=criterion_ae,
        optimizer=optimizer_ae,
        epochs=AE_EPOCHS
    )

    plot_history(
        ae_loss,
        "Autoencoder3D Reconstruction Loss",
        "MSE Loss",
        "autoencoder3d_reconstruction_loss.png"
    )

    visualize_autoencoder(
        model_ae,
        ae_dataset,
        "autoencoder3d_reconstruction_example.png"
    )

    torch.save(model_ae.state_dict(), "autoencoder3d_epilepsy_fusion.pth")
    print("Saved: autoencoder3d_epilepsy_fusion.pth")

    print("\nГотово.")
    print("Създадени файлове:")
    print("- resnet3d_epilepsy_fusion.pth")
    print("- autoencoder3d_epilepsy_fusion.pth")
    print("- resnet3d_training_loss.png")
    print("- resnet3d_training_accuracy.png")
    print("- autoencoder3d_reconstruction_loss.png")
    print("- autoencoder3d_reconstruction_example.png")


if __name__ == "__main__":
    main()