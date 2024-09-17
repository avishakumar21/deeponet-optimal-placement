import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from opnn_avisha import opnn
from dataset_prep import get_paths, TransducerDataset

def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for branch1_input, branch2_input, trunk_input, labels in dataloader:
            # Move inputs and labels to the GPU
            branch1_input = branch1_input.to(device)
            branch2_input = branch2_input.to(device)
            trunk_input = trunk_input.to(device)
            labels = labels.to(device)
            
            # Calculate loss
            loss = model.loss(branch1_input, branch2_input, trunk_input, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        #avg_loss = total_loss / len(dataloader)
        avg_loss = total_loss / (len(dataloader)*dataloader.dataset.sim_height*dataloader.dataset.sim_width)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model dimensions for the branches and trunk
    bz = 10  # Batch size

    # Define the architecture of the branches and trunk network
    branch1_dim = [668*275, 100, 100, 64]  # Geometry branch dimensions (flattened image input followed by layers)
    branch2_dim = [2, 64, 32, 16]  # Source location branch
    trunk_dim = [2, 100, 64, 32]  # Trunk network (grid coordinates)

    # Define geometry_dim and output_dim based on your data
    geometry_dim = (275, 668)  # Image dimensions (height, width)
    output_dim = (162 * 512)  # Simulation dimensions (pressure map height and width) #162 * 512

    # Initialize model and move it to the device (GPU/CPU)
    model = opnn(branch1_dim, branch2_dim, trunk_dim, geometry_dim, output_dim).to(device)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=0.0001)

    # Prepare data
    images_path, masks_path, simulation_path = get_paths('data/')  # Change this path
    dataset = TransducerDataset(images_path, simulation_path, loading_method='individual')
    dataloader = DataLoader(dataset, batch_size=bz, shuffle=True, num_workers=2)

    # Train the model
    train_model(model, dataloader, optimizer, device, num_epochs=1000)


