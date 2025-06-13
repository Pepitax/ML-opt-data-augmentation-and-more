import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import random
import os

# ------------------------------------------------------------
# Constants and configuration
# ------------------------------------------------------------
DATA_DIR = "./data"
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_TRAIN_SAMPLES = 1000 #200 #1000

# ------------------------------------------------------------
# Device setup
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(DEVICE.index or 0)})")
    print(f"{torch.cuda.device_count()} GPU(s) available:")
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Using device: CPU")

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# Dataset statistics helper
# ------------------------------------------------------------
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=NUM_TRAIN_SAMPLES, shuffle=False)
    data = next(iter(loader))[0]
    mean = data.mean(dim=[0, 2, 3])
    std = data.std(dim=[0, 2, 3])
    return mean, std

# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------
# Run training for multiple seeds
# ------------------------------------------------------------
accuracies = []

for SEED in range(1, 11):
    print(f"\n======== Seed {SEED} ========")
    seed_everything(SEED)

    transform_no_norm = transforms.ToTensor()
    full_trainset_raw = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_no_norm)
    subset_indices = np.random.choice(len(full_trainset_raw), NUM_TRAIN_SAMPLES, replace=False)
    trainset_raw = Subset(full_trainset_raw, subset_indices)

    computed_mean, computed_std = compute_mean_std(trainset_raw)
    transform_tensor_input = transforms.Normalize(computed_mean.tolist(), computed_std.tolist())

    class AugmentedAllDataset(Dataset):
        def __init__(self, base_subset):
            self.data = []
            for img, label in base_subset:
                self.data.append((img, label))
                self.data.append((transforms.functional.hflip(img), label))

                img_pil = transforms.ToPILImage()(img)
                cropped = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor()
                ])(img_pil)
                self.data.append((cropped, label))
                self.data.append((transforms.functional.hflip(cropped), label))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img, label = self.data[idx]
            return transform_tensor_input(img), label

    trainset = AugmentedAllDataset(trainset_raw)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(computed_mean.tolist(), computed_std.tolist())
    ])
    testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


    model = CIFAR10CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train_one_epoch(epoch):
        model.train()
        for images, targets in trainloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct, total = 0, 0
        for images, targets in testloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = correct / total
        return acc

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(epoch)

    acc = evaluate()
    accuracies.append(acc)
    print(f"Final Test Accuracy: {acc:.4f}")

print("\n======== Summary =========")
print("Accuracies for seeds 1 to 10:")
print(accuracies)
print(f"Mean: {np.mean(accuracies):.4f} | Std: {np.std(accuracies):.4f}")

script_name = os.path.basename(__file__)
results_filename = "results_" + script_name.replace(".py", "") + f"_{NUM_TRAIN_SAMPLES}.txt"

results_path = os.path.join(os.path.dirname(__file__), results_filename)

with open(results_path, "w") as f:
    f.write(" ".join(f"{acc:.6f}" for acc in accuracies))

print(f"Results written to: {results_path}")
