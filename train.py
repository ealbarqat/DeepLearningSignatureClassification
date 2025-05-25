# Let's import all the tools we need
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# This class helps us load our signature data
class SignatureDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, max_length=500):
        # Store the paths to our signature files and their labels
        self.file_paths = file_paths
        self.labels = labels
        # We'll make all signatures the same length (500 points)
        self.max_length = max_length
        
    def __len__(self):
        # Tell PyTorch how many signatures we have
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Get one signature and its label
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Read the signature file (skip the first line as it's just headers)
        data = pd.read_csv(file_path, skiprows=1, header=None)
        
        # Convert the signature points to numbers
        coords = []
        for coord_pair in data[0]:
            x, y = map(float, coord_pair.split())
            coords.append([x, y])
        
        # Turn our list into a numpy array
        coords = np.array(coords, dtype=np.float32)
        
        # Make all signatures the same length
        if len(coords) > self.max_length:
            # If signature is too long, cut it off
            coords = coords[:self.max_length]
        elif len(coords) < self.max_length:
            # If signature is too short, add zeros at the end
            padding = np.zeros((self.max_length - len(coords), 2), dtype=np.float32)
            coords = np.vstack([coords, padding])
        
        # Convert to PyTorch tensor
        coords = torch.from_numpy(coords)
        
        return coords, label

# This is our neural network model
class SignatureClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_classes=4):
        super(SignatureClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer - good for learning patterns in sequences
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Final layers to make our prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Helps prevent overfitting
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Set up initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Run the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the final prediction
        out = self.fc(out[:, -1, :])
        return out

# This function helps us find all our signature files
def get_data_paths(root_dir):
    file_paths = []
    labels = []
    # Map each type of signature to a number
    class_mapping = {'human': 0, 'gan': 1, 'sdt': 2, 'vae': 3}
    
    # Go through each type of signature
    for class_name, label in class_mapping.items():
        class_dir = os.path.join(root_dir, class_name)
        # Get all CSV files in this folder
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.csv'):
                file_paths.append(os.path.join(class_dir, file_name))
                labels.append(label)
    
    return file_paths, labels

# This function trains our model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    
    # Train for the specified number of epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Go through our training data
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear previous gradients
            optimizer.zero_grad()
            # Get model predictions
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Backpropagate
            loss.backward()
            # Update weights
            optimizer.step()
            
            # Keep track of how well we're doing
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Check how well we're doing on validation data
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Print our progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save our best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

# This is where everything starts
def main():
    # Set random seeds so we get the same results each time
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if we can use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get all our signature files
    root_dir = 'signatures'
    file_paths, labels = get_data_paths(root_dir)
    
    # Split our data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create our datasets
    train_dataset = SignatureDataset(train_paths, train_labels, max_length=500)
    val_dataset = SignatureDataset(val_paths, val_labels, max_length=500)
    
    # Create data loaders to feed data to our model
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create our model
    model = SignatureClassifier().to(device)
    
    # Set up our loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train our model
    num_epochs = 50
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

# Run the main function when we start the script
if __name__ == '__main__':
    main() 