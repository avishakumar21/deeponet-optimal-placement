import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os

from opnn import opnn
from dataset_prep import get_paths, TransducerDataset
from utils import log_loss, save_loss_to_dated_file, plot_logs,plot_prediction,store_model
from utils import ensure_directory_exists,get_time_YYYYMMDDHH
import argparse

#!!! No slash at the end of the path
# DATA_PATH = r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\mini deeponet dataset'
# result_folder = r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\deeponet low resolution code 2\result'
DATA_PATH ="data"
RESULT_FOLDER = "result\result" #change folder name
epochs = 3

class Trainer:
    def __init__(self, model, optimizer, device, num_epochs=1000):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
        self.test_loss = None

    def set_result_path (self, result_path):
        self.result_folder = result_path
        self.val_log_path = os.path.join(self.result_folder, 'loss_log_train.txt')
        self.train_log_path = os.path.join(self.result_folder, 'loss_log_val.txt')
        print(f"Result of this Training will be stored at {result_path}")

    def train_one_epoch(self, dataloader):
        self.model.train() 
        total_loss = 0.0 #[]
        total_samples = 0
        for branch1_input, branch2_input, trunk_input, labels in dataloader:
            # Move inputs and labels to the GPU
            branch1_input = branch1_input.to(self.device)
            branch2_input = branch2_input.to(self.device)
            trunk_input = trunk_input.to(self.device)
            labels = labels.to(self.device)
            
            # Calculate loss
            loss = self.model.loss(branch1_input, branch2_input, trunk_input, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            #total_loss.append(loss.item())

            #count sample
            total_samples += labels.numel()

            break
        avg_loss = total_loss / total_samples
        #norm total_loss
        self.train_losses.append(avg_loss)
        return avg_loss

    def val_one_epoch(self, dataloader_validation):
        self.model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        total_samples = 0 
        with torch.no_grad():
            for branch1_input, branch2_input, trunk_input, labels in dataloader_validation:
                branch1_input = branch1_input.to(self.device)
                branch2_input = branch2_input.to(self.device)
                trunk_input = trunk_input.to(self.device)
                labels = labels.to(self.device)

                val_loss = self.model.loss(branch1_input, branch2_input, trunk_input, labels)
                total_val_loss += val_loss.item()

                total_samples += labels.numel()
  
        avg_val_loss = total_val_loss / total_samples
        self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def test(self, dataloader_test, epochs = 0):
        self.model.eval()  # Set the model to evaluation mode
        total_test_loss = 0.0

        with torch.no_grad():
            for batch, (branch1_input, branch2_input, trunk_input, labels) in enumerate(dataloader_test):
                branch1_input = branch1_input.to(self.device)
                branch2_input = branch2_input.to(self.device)
                trunk_input = trunk_input.to(self.device)
                labels = labels.to(self.device)

                test_loss = self.model.loss(branch1_input, branch2_input, trunk_input, labels)
                total_test_loss += test_loss.item()
                #plot sample prediction:
                prediction = self.model(branch1_input, branch2_input, trunk_input)
                prediction = torch.mean(prediction, dim=1)
                plot_prediction(branch1_input.cpu(), labels.cpu(), prediction.cpu(), batch, result_folder=self.result_folder)

        avg_test_loss = total_test_loss / (len(dataloader_test) * dataloader_test.dataset.sim_height * dataloader_test.dataset.sim_width)
        self.test_loss = avg_test_loss
        return avg_test_loss

    def train(self, dataloader, dataloader_validation, dataloader_test=None):
        ensure_directory_exists(self.result_folder)
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(dataloader)
            val_loss = self.val_one_epoch(dataloader_validation)
            
            log_loss(train_loss, temp_file=self.train_log_path)
            log_loss(val_loss, self.val_log_path)
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # every 5000, some sample viz  

        # After training, perform testing
        if dataloader_test is not None:
            test_loss = self.test(dataloader_test)
            print(f"Test Loss: {test_loss:.4f}")

     
        store_model(self.model,self.optimizer, epoch, self.result_folder)
        return self.model
    
def load_data(data_path, bz):
    images_path, masks_path, simulation_path = get_paths(data_path)  # Change this path
    print('prepared data, model, optimizer')

    # train_val split
    train_val_ratio = 0.8
    split_idx = int(len(images_path)*train_val_ratio)-1
    dataset_train = TransducerDataset(images_path[:split_idx], simulation_path[:split_idx*8], loading_method='individual')
    dataset_valid = TransducerDataset(images_path[split_idx:-2], simulation_path[split_idx*8:-16], loading_method='individual')
    dataset_test = TransducerDataset(images_path[-2:], simulation_path[-16:], loading_method='individual')
    dataloader_train = DataLoader(dataset_train, batch_size=bz, shuffle=True, num_workers=2)
    dataloader_valid = DataLoader(dataset_valid, batch_size=bz, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(dataset_test, batch_size=bz, shuffle=True, num_workers=2)
    return dataloader_train, dataloader_valid,dataloader_test

def main(bz, num_epochs=100, result_folder = RESULT_FOLDER, folder_description = ""):
    #Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_folder = result_folder+get_time_YYYYMMDDHH()+'_'+folder_description
    ensure_directory_exists(result_folder)

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
    dataloader_train, dataloader_valid,dataloader_test = load_data(DATA_PATH, bz)

    # Train the model
    trainer = Trainer(model, optimizer, device, num_epochs)
    trainer.set_result_path(result_folder)
    model = trainer.train(dataloader_train, dataloader_valid, dataloader_test)

    #Plot Losses
    file_paths = [trainer.train_log_path,trainer.val_log_path]
    plot_logs(file_paths, output_image=os.path.join(trainer.result_folder, "loss_plot.png"))

    #Package Experiment data into a new folder
if __name__ == "__main__":
    # Add an optional exp description argument
    parser = argparse.ArgumentParser(description="Experiment id or brief description, no space or slash allowed. Good Example: high_resolution_1.")
    parser.add_argument('exp_description', type=str, nargs='?', default="", help='Optional Experiment description. Good Example: high_resolution_1.')
    args = parser.parse_args()

    # Call the main function and pass the argument
    main(bz=3,num_epochs=epochs, result_folder=RESULT_FOLDER, folder_description=args.exp_description)
    