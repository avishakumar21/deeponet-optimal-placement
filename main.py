import torch
from torch.utils.data import DataLoader,Subset
from torch.optim import Adam
import os

#from opnn_transformer import opnn
from opnn import opnn

from dataset_prep import get_paths, TransducerDataset
from utils import log_loss, save_loss_to_dated_file, plot_logs,plot_prediction,store_model
from utils import ensure_directory_exists,get_time_YYYYMMDDHH
import argparse
from torch.optim.lr_scheduler import StepLR


#!!! No slash at the end of the path
DATA_PATH = r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\deeponet dataset 1000'
RESULT_FOLDER = r'C:\Users\akumar80\Documents\Avisha Kumar Lab Work\deeponet result 1000'
#DATA_PATH ="data"
#RESULT_FOLDER = "result/result" #change folder name
loading_method = 'loc_7' #individual

epochs = 4000 #total epochs to run
VIZ_epoch_period = 200 #visualize sample validation set image every VIZ_epoch_period
BATCHSIZE = 16 
STEP_SIZE = 200 

EXPECTED_IMG_SIZE = (162, 512)
EXPECTED_SIM_SIZE = (162, 512)

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

            #break
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
        total_samples = 0 

        with torch.no_grad():
            for batch, (branch1_input, branch2_input, trunk_input, labels) in enumerate(dataloader_test):
                branch1_input = branch1_input.to(self.device)
                branch2_input = branch2_input.to(self.device)
                trunk_input = trunk_input.to(self.device)
                labels = labels.to(self.device)

                test_loss = self.model.loss(branch1_input, branch2_input, trunk_input, labels)
                total_test_loss += test_loss.item()
                total_samples += labels.numel()

                #plot sample prediction:
                # prediction = self.model(branch1_input, branch2_input, trunk_input)
                # prediction = torch.mean(prediction, dim=1)
                # plot_prediction(branch1_input.cpu(), labels.cpu(), prediction.cpu(), batch, result_folder=self.result_folder)

        #self.visualize_prediction(dataloader_test, comment = 'testset',subset=False)

        avg_test_loss = total_test_loss / total_samples
        self.test_loss = avg_test_loss
        return avg_test_loss
    
    def visualize_prediction(self, dataloader, comment = '', subset = True):
        if subset:
            # Select a subset of indices, e.g., the first 100 samples
            subset_indices = list(range(BATCHSIZE))
            # Create a subset using the Subset class
            subset_dataset = Subset(dataloader.dataset, subset_indices)
            # Create a new DataLoader with this subset
            dataloader = DataLoader(subset_dataset, batch_size=BATCHSIZE, shuffle=False)

        #epoch can be int to represent epoch / string 'end' to represent end to trainning
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch, (branch1_input, branch2_input, trunk_input, labels) in enumerate(dataloader):
                prediction = self.model(branch1_input.to(self.device), branch2_input.to(self.device), trunk_input.to(self.device))
                prediction = torch.mean(prediction, dim=1)
                #plot_prediction(branch1_input.cpu(), labels.cpu(), prediction.cpu(), batch, comment = comment, result_folder=self.result_folder)
        
        return True


    def train(self, dataloader, dataloader_validation, dataloader_test, scheduler):
        # Check Result Directory
        ensure_directory_exists(self.result_folder)

        # Train + Validate Model
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(dataloader)
            scheduler.step()
            val_loss = self.val_one_epoch(dataloader_validation)
            
            log_loss(train_loss, temp_file=self.train_log_path)
            log_loss(val_loss, self.val_log_path)
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # every X epoch, some sample viz 
            if epoch % VIZ_epoch_period == 0:
                self.visualize_prediction(dataloader_validation, comment = f'val_ep{epoch}',subset=True)

        # Test Model
        if dataloader_test is not None:
            test_loss = self.test(dataloader_test)
            print(f"Test Loss: {test_loss:.4f}")

        # Store model
        store_model(self.model,self.optimizer, epoch, self.result_folder)
        return self.model
    

def load_data_by_split(data_path, bz, shuffle = True):
    print('-'*15, 'DATA READIN BY SPLIT', '-'*15)
    split_path_dict = {}
    for split_name in ['train','val','test']:
        split_data_path=os.path.join(data_path, '{data_type}',split_name)
        images_path,simulation_path = get_paths(split_data_path)
        dataset_ = TransducerDataset(images_path, simulation_path, loading_method=loading_method)
        dataloader_ = DataLoader(dataset_, batch_size=bz, shuffle=shuffle, num_workers=2)
        split_path_dict[split_name] = dataloader_

    return list(split_path_dict.values())

def main(bz, num_epochs=100, result_folder = RESULT_FOLDER, folder_description = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify Unique Directories for result
    print('-'*15, 'CHECK RESULT DIRECTORY', '-'*15)
    result_folder = result_folder+get_time_YYYYMMDDHH()+'_'+folder_description
    ensure_directory_exists(result_folder)

    # Define the architecture of the branches and trunk network
    #branch1_dim = [EXPECTED_IMG_SIZE[1]*EXPECTED_IMG_SIZE[0], 100, 100, 64]  # Geometry branch dimensions (flattened image input followed by layers)
    branch2_dim = [2, 32, 32, 64]  # Source location branch
    trunk_dim = [2, 100, 100, 64]  # Trunk network (grid coordinates)

    # Define geometry_dim and output_dim based on your data
    geometry_dim = EXPECTED_IMG_SIZE  # Image dimensions (height, width)
    #output_dim = EXPECTED_SIM_SIZE[0] * EXPECTED_SIM_SIZE[1]  # Simulation dimensions (pressure map height and width) #162 * 512

    # Initialize model and move it to the device (GPU/CPU)
    model = opnn(branch2_dim, trunk_dim, geometry_dim).to(device) # for CNN
    # model = opnn(branch2_dim, trunk_dim, geometry_dim, patch_size = 9).to(device) # for transformer



    # Initialize optimizer
    #optimizer = Adam(model.parameters(), lr=0.0001) 
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5)  # Reduce LR every 500 steps


    # Prepare data
    dataloader_train, dataloader_valid,dataloader_test = load_data_by_split(DATA_PATH, bz)

    # Train the model
    print('-'*15, 'TRAIN', '-'*15)
    trainer = Trainer(model, optimizer, device, num_epochs)
    trainer.set_result_path(result_folder)
    model = trainer.train(dataloader_train, dataloader_valid, dataloader_test, scheduler)

    #Plot Losses
    file_paths = [trainer.train_log_path,trainer.val_log_path]
    plot_logs(file_paths, output_image=os.path.join(trainer.result_folder, "loss_plot.png"))

    
if __name__ == "__main__":
    # Add an optional exp description argument
    parser = argparse.ArgumentParser(description="Experiment id or brief description, no space or slash allowed. Good Example: high_resolution_1.")
    parser.add_argument('exp_description', type=str, nargs='?', default="", help='Optional Experiment description. Good Example: high_resolution_1.')
    args = parser.parse_args()

    # Call the main function and pass the argument
    main(bz=BATCHSIZE,num_epochs=epochs, result_folder=RESULT_FOLDER, folder_description=args.exp_description)
    