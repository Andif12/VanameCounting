import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

# CONFIG
DATASET_DIR = "dataset_verifier"
MODEL_OUT = "model/efficientnet_lite_verifier.pth"

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
IMG_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRANSFORMS
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# DATASET
train_ds = datasets.ImageFolder(Path(DATASET_DIR) / "train", transform=train_tf)
val_ds   = datasets.ImageFolder(Path(DATASET_DIR) / "val", transform=val_tf)

print("Class mapping:", train_ds.class_to_idx)
# HARUS: {'benur_invalid': 0, 'benur_valid': 1}

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# MODEL
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# TRAIN LOOP
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.3f} | Val Acc: {acc:.2f}%")

# SAVE
Path("model").mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
print("\n✅ Verifier model saved:", MODEL_OUT)