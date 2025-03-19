import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class ApneaMelDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads preprocessed segments and labels.
        """
        segment_data = torch.load(self.data[idx], weights_only=True)
        segment = segment_data["mel"]
        label = segment_data["label"]
        return segment, label

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, input_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 32, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = torch.mean(x, dim=-1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# Train and evaluate the CNN model
def train_cnn(epochs=20, batch_size=32, patience=3):
    # Datasets and Loaders
    train_dataset = ApneaMelDataset(data_dir="../preprocessed_apnea_mel/train")
    val_dataset = ApneaMelDataset(data_dir="../preprocessed_apnea_mel/val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Initialize model, criterion, and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")
        for data, target in progress_bar:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({"Loss": train_loss / len(train_loader)})

        # Validation loop
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation")
            for data, target in progress_bar:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                val_loss += criterion(output, target).item()
                preds = torch.argmax(output, dim=1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            progress_bar.set_postfix({"Val Loss": val_loss / len(val_loader)})

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(classification_report(y_true, y_pred, target_names=["No ObstructiveApnea", "ObstructiveApnea"]))

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "../output/cnn_obstructive_apnea.pth")
            print("Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def predict_and_save_logits(model_path, data_dir, output_path, batch_size=128):
    # Load model
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dataset and Loader (no shuffling for consistency in ensemble learning)
    dataset = ApneaMelDataset(data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Predicting on {data_dir}")
        for data, target in progress_bar:
            data = data.to(device)
            target = target.to(device)
            
            logits = model(data)
            all_logits.append(logits.cpu())
            all_labels.append(target.cpu())
    
    # Save logits and labels
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"Logits shape: {all_logits.shape}")
    torch.save({"logits": all_logits, "labels": all_labels}, output_path)
    print(f"Saved logits and labels to {output_path}")

# Paths to WAV and RML folders
train_cnn(epochs=30, batch_size=16)

# Run inference and save logits for train and test sets
predict_and_save_logits("../output/cnn_obstructive_apnea.pth", "../preprocessed_apnea_mel/train", "../output/cnn_train_logits.pth")
predict_and_save_logits("../output/cnn_obstructive_apnea.pth", "../preprocessed_apnea_mel/val", "../output/cnn_val_logits.pth")