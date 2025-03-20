import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Data Loading & Preparation
# -----------------------------
# Load logits and labels (replace with your actual file paths)
crnn_train = torch.load("my_api_project/outputs/crnn_train_logits.pth")
crnn_val   = torch.load("my_api_project/outputs/crnn_val_logits.pth")
lstm_train = torch.load("my_api_project/outputs/lstm_train_logits.pth")
lstm_val   = torch.load("my_api_project/outputs/lstm_val_logits.pth")
vggish_train = torch.load("my_api_project/outputs/vggish_train_logits.pth")
vggish_val   = torch.load("my_api_project/outputs/vggish_val_logits.pth")

crnn_logits_train = crnn_train["logits"]
crnn_logits_val   = crnn_val["logits"]
lstm_logits_train = lstm_train["logits"]
lstm_logits_val   = lstm_val["logits"]
vggish_logits_train = vggish_train["logits"]
vggish_logits_val   = vggish_val["logits"]

train_labels = crnn_train["labels"]
val_labels   = crnn_val["labels"]

def logits_to_positive_probs(logits):
    """Convert logits [batch_size, 2] to positive-class probabilities."""
    probs = F.softmax(logits, dim=1)
    return probs[:, 1]

# Convert logits to probabilities for each model
crnn_probs_train   = logits_to_positive_probs(crnn_logits_train)
lstm_probs_train   = logits_to_positive_probs(lstm_logits_train)
vggish_probs_train = logits_to_positive_probs(vggish_logits_train)

crnn_probs_val   = logits_to_positive_probs(crnn_logits_val)
lstm_probs_val   = logits_to_positive_probs(lstm_logits_val)
vggish_probs_val = logits_to_positive_probs(vggish_logits_val)

# Stack predictions to form ensemble inputs
ensemble_train_inputs = torch.stack((crnn_probs_train, lstm_probs_train, vggish_probs_train), dim=1)
ensemble_val_inputs   = torch.stack((crnn_probs_val, lstm_probs_val, vggish_probs_val), dim=1)

# Convert labels to float and add dimension for BCE loss
train_labels_tensor = train_labels.float().unsqueeze(1)
val_labels_tensor   = val_labels.float().unsqueeze(1)

# Create datasets and DataLoaders with multiple workers for efficiency
train_dataset = TensorDataset(ensemble_train_inputs, train_labels_tensor)
val_dataset   = TensorDataset(ensemble_val_inputs, val_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# -----------------------------
# Define the Meta-Learner Model
# -----------------------------
class GatingNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output a single logit

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Outputs probability in [0,1]
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gating_net = GatingNetwork(input_dim=3, hidden_dim=8).to(device)
print(gating_net)

# -----------------------------
# Training Function
# -----------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    avg_loss = running_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds).numpy().squeeze()
    all_labels = torch.cat(all_labels).numpy().squeeze()
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["No Apnea", "Apnea"])
    return avg_loss, acc, report

# -----------------------------
# Set Up Training
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(gating_net.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    train_loss = train_epoch(gating_net, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_report = evaluate(gating_net, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
    print(val_report)
    
# -----------------------------
# Save the Trained Model
# -----------------------------
torch.save(gating_net.state_dict(), "final_meta_learner.pth")
print("Final Meta-Learner saved!")