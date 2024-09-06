import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from opnn import opnn

def train_model(model, dataloader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for branch1_input, branch2_input, trunk_input, labels in dataloader:
            # Forward pass and compute loss using the model's loss method
            loss = model.loss(branch1_input, branch2_input, trunk_input, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Define the model dimensions for the branches and trunk
    branch1_dim = [10, 64, 64]   # Example dimensions for branch 1
    branch2_dim = [10, 64, 64]   # Example dimensions for branch 2
    trunk_dim = [20, 64, 64]     # Example dimensions for the trunk

    # Initialize the model
    model = opnn(branch1_dim, branch2_dim, trunk_dim)

    # Define an optimizer (e.g., Adam)
    optimizer = Adam(model.parameters(), lr=0.001)

    # Create a DataLoader (assuming `dataloader` is defined)
    dataloader = DataLoader(...)  # Replace with actual DataLoader

    # Train the model
    train_model(model, dataloader, optimizer, num_epochs=10)