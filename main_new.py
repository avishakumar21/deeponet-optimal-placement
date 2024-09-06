import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from opnn import opnn
from dataset_prep import get_paths,TransducerDataset



def train_model(model, dataloader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for branch1_input, branch2_input, trunk_input, labels in dataloader:
            print(total_loss)
            loss = model.loss(branch1_input, branch2_input, trunk_input, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Define the model dimensions for the branches and trunk
    bz = 10
    #torch.Size([10, 3, 195, 610]) torch.Size([10, 2]) torch.Size([10, 195, 610, 2]) torch.Size([10, 162, 512])
    branch1_dim = [bz, 3, 195, 610] # Example dimensions for branch 1
    branch2_dim = [bz, 2]   # Example dimensions for branch 2
    trunk_dim = [bz, 195, 610, 2]     # Example dimensions for the trunk

    #model
    model = opnn(branch1_dim, branch2_dim, trunk_dim)

    #optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    #data loader
    images_path, masks_path, simulation_path = get_paths('data')
    dataset = TransducerDataset(images_path, simulation_path, loading_method = 'individual')
    dataloader = DataLoader(dataset, batch_size=bz, shuffle=True, num_workers=2)
    

    # Train the model
    train_model(model, dataloader, optimizer, num_epochs=10)