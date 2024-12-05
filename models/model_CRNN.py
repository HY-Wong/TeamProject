import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

train_data_path = 'train.npz'
val_data_path = 'val.npz'
num_classes = 4

# Load training data
train_data = np.load(train_data_path)
train_spectrograms = train_data['spectrograms']
train_labels = train_data['labels']

# Load validation data
val_data = np.load(val_data_path)
val_spectrograms = val_data['spectrograms']
val_labels = val_data['labels']

device = device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.spectrograms = torch.tensor(spectrograms, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]
    
# Create DataLoaders
train_dataset = SpectrogramDataset(train_spectrograms, train_labels)
val_dataset = SpectrogramDataset(val_spectrograms, val_labels)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Define CRNN model architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn(self.conv(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.rnn = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # (batch_size, 1, time_steps, mel_bins)
        x = x.permute(0, 2, 3, 1)
        x = self.bn(x) # normalize per mel_bins
        x = x.permute(0, 3, 2, 1)

        # CNN feature extractor
        # (batch size, channels, time, frequency)
        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        # Aggregate in frequency axis
        # (batch size, channels, time)
        x = torch.mean(x, dim=3)
        
        # (batch size, time, channels)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # LSTM
        # (batch size, channels)
        x, _ = self.rnn(x)
        x = x[:, -1, :]

        # (batch size, num_classes)
        x = self.fc2(x)
        return x

# Initialize model, criterion, and optimizer
model = CRNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(20):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for spectrograms, labels in train_loader:
        spectrograms = spectrograms.unsqueeze(1) # channel dimension
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(spectrograms)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # Calculate training accuracy
        _, predicted = torch.max(output, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train

    # Evaluate the model on the validation set
    model.eval()
    total_vloss = 0.0
    correct_val = 0 
    total_val = 0 

    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = spectrograms.unsqueeze(1) # channel dimension
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            output = model(spectrograms)
            loss = criterion(output, labels)
            
            total_vloss += loss.item()
            # Calculate validation accuracy
            _, predicted = torch.max(output, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = correct_val / total_val

    print(f'Epoch {epoch+1:2d}, Loss: {total_loss / len(train_loader):.4f}, VLoss: {total_vloss / len(val_loader):.4f} Accuracy: {train_accuracy:.4f}, VAccuracy: {val_accuracy:.4f}')

# Evaluate the model on the validation set
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for spectrograms, labels in val_loader:
        spectrograms = spectrograms.unsqueeze(1) # channel dimension
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        output = model(spectrograms)
        _, predicted = torch.max(output, 1)

        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

print('-'*80)
print(classification_report(y_true, y_pred, labels=np.arange(4), target_names=['ObstructiveApnea', 'Hypopnea', 'CentralApnea', 'MixedApnea']))
print('-'*80)
print(confusion_matrix(y_true, y_pred))
print('-'*80)