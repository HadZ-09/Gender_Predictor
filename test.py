
# train.py

import os
import glob
import random
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models


# ---------------- CONFIGURATION ----------------

DATA_DIR = "./UTKFace"
SAVE_DIR = "./weights"
MODEL_PATH = os.path.join(SAVE_DIR, "gender_resnet18.pt")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------- REPRODUCIBILITY ----------------

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------- TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ---------------- DATASET ----------------

class UTKFaceGender(Dataset):

    def __init__(self, root, transform=None):

        self.transform = transform
        self.samples = []

        paths = sorted(
            glob.glob(
                os.path.join(root, "**", "*.jpg"),
                recursive=True
            )
        )

        for path in paths:

            filename = os.path.basename(path)
            parts = filename.split("_")

            try:
                label = int(parts[1])

                if label in (0, 1):
                    self.samples.append((path, label))

            except (IndexError, ValueError):
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------- LOAD DATA ----------------

train_base = UTKFaceGender(
    DATA_DIR,
    transform=train_transform
)

eval_base = UTKFaceGender(
    DATA_DIR,
    transform=eval_transform
)

if len(train_base) == 0:
    raise RuntimeError(
        "No valid UTKFace images found. Check DATA_DIR."
    )

assert len(train_base) == len(eval_base)

n = len(train_base)

generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(n, generator=generator).tolist()

n_train = int(0.70 * n)
n_val = int(0.15 * n)

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train + n_val]

train_dataset = Subset(train_base, train_indices)
val_dataset = Subset(eval_base, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Total images:", n)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Device:", DEVICE)


# ---------------- MODEL ----------------

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(DEVICE)


# ---------------- LOSS AND OPTIMIZER ----------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ---------------- TRAINING FUNCTION ----------------

def run_epoch(loader, training=True):

    model.train(training)

    total_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += images.size(0)

    return total_loss / total, correct / total


# ---------------- TRAIN LOOP ----------------

best_val_acc = 0.0

for epoch in range(EPOCHS):

    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    scheduler.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc*100:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_acc*100:.2f}%"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": {
                    0: "Male",
                    1: "Female"
                },
                "best_val_accuracy": best_val_acc
            },
            MODEL_PATH
        )

        print("Best model saved!")

print("\nTraining completed.")
print("Best validation accuracy:", best_val_acc)
