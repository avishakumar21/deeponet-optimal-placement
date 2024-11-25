import torch
import time
from opnn import opnn
from dataset_prep import get_paths, TransducerDataset
from torch.utils.data import DataLoader
import os

# Model configuration
EXPECTED_IMG_SIZE = (162, 512)
branch2_dim = [2, 32, 32, 64]  # Source location branch
trunk_dim = [2, 100, 100, 64]  # Trunk network (grid coordinates)
geometry_dim = EXPECTED_IMG_SIZE

# File paths
DATA_PATH_IMAGES = r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\deeponet dataset 1000\masks\test'
DATA_PATH_SIMULATIONS = r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\deeponet dataset 1000\simulation_outputs\test'
CHECKPOINT_PATH = r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\results\model_checkpoint.pth'

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = opnn(branch2_dim, trunk_dim, geometry_dim).to(device)

# Load the checkpoint file
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Extract the model's state_dict and load it into the model
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Prepare the dataset
image_paths, _ = get_paths(DATA_PATH_IMAGES)  # Get images from the mask path
_, simulation_paths = get_paths(DATA_PATH_SIMULATIONS)  # Get simulations from the correct path

print(f"Found {len(image_paths)} image files.")
print(f"Found {len(simulation_paths)} simulation files.")

if len(image_paths) == 0 or len(simulation_paths) == 0:
    raise ValueError("No image or simulation files found. Please check the dataset structure or file extensions.")

# Create the dataset and dataloader
test_dataset = TransducerDataset(image_paths, simulation_paths, loading_method='individual', device=device)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Testing a single image
for i, (image, transducer_locs, locs, simulations) in enumerate(test_loader):
    print(f"Processing image {i+1}")
    
    # Move data to the device
    image = image.to(device)
    transducer_locs = transducer_locs.to(device)
    locs = locs.to(device)
    
    # Time the inference
    start_time = time.time()
    with torch.no_grad():
        prediction = model(image, transducer_locs, locs)
    end_time = time.time()

    # Print timing information
    print(f"Time taken for inference on image {i+1}: {end_time - start_time:.6f} seconds")

    # Optional: Display prediction shape
    print(f"Prediction shape for image {i+1}: {prediction.shape}")

    # Break after processing one image (remove this if testing all images)
    break
