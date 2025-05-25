# You can import any module, model class, etc.
# We will import the `load_and_predict()` function below to assess your assignment.

# Let's import the tools we need
import os
import torch
import torch.nn as nn
import pandas as pd
from train import SignatureClassifier

def load_and_predict(directory, model_file):
    # Check if we can use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create our model
    model = SignatureClassifier()
    
    # Load our trained model
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()  # Tell the model we're testing, not training
    
    # Dictionary to store our predictions
    predictions = {}
    
    # Go through each type of signature
    for class_name in ['human', 'gan', 'sdt', 'vae']:
        class_dir = os.path.join(directory, class_name)
        
        # Look at each signature file in this folder
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(class_dir, file_name)
                
                # Read the signature file
                data = pd.read_csv(file_path, skiprows=1, header=None)
                
                # Convert the signature points to numbers
                coords = []
                for coord_pair in data[0]:
                    x, y = map(float, coord_pair.split())
                    coords.append([x, y])
                
                # Make the signature the right length (500 points)
                coords = torch.tensor(coords, dtype=torch.float32)
                if len(coords) > 500:
                    coords = coords[:500]
                elif len(coords) < 500:
                    padding = torch.zeros((500 - len(coords), 2), dtype=torch.float32)
                    coords = torch.cat([coords, padding])
                
                # Add batch dimension and move to device
                coords = coords.unsqueeze(0).to(device)
                
                # Get the model's prediction
                with torch.no_grad():
                    outputs = model(coords)
                    _, predicted = torch.max(outputs, 1)
                
                # Store the prediction
                predictions[file_path] = predicted.item()
    
    return predictions

