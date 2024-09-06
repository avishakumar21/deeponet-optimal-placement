#File for Hepius project dataset
import glob
import numpy as np
import torch
import os
#import albumentations as A
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
"""
1. ultrasound Image Read in as RGB / Gray
2. Transducer Location: 2 orientation: a) object x location x xy-coordinate; b) object x location x xy-coordinate
3. Result map

"""

def get_paths(root_path):
    # given path to dataset (contain sub directory images, masks, simulation_outputs)
    images_path = glob.glob(f"{root_path}/images/*")
    images_path.sort()
    masks_path = glob.glob(f"{root_path}/masks/*")
    masks_path.sort()
    simulation_path = glob.glob(os.path.join(f"{root_path}/simulation_outputs/", '**', 'maximum_pressure_distribution*.png'), recursive=True)
    simulation_path.sort()
   

    return images_path, masks_path, simulation_path, 


class TransducerDataset(Dataset):
    def __init__(
        self, 
        image_path, 
        simulation_path, 
        image_transforms = True,
        loading_method = 'individual'
    ):
        """
        image_transforms: bool: decide if apply internally defined transform to image
        loading_method: str, inidividual / group.
                - inidividual: treat each transducer location - image pair as one sample
                - group: treat each image and corresponding 8 transducer location as one object
        
        """
        self.image_paths = image_path
        self.simulation_paths = simulation_path
        self.image_transforms = image_transforms
        # self.tfms = tfms
        # self.norm_tfms = norm_tfms
        self.loading_method = loading_method


        self.width = 610
        self.height = 195
        
        #Transducer location
        transducer_locs=np.array([[0]*8,[x-100 for x in [150, 220, 295, 370, 440,515, 585, 660]]]) 
        # CHECK: transducer_locs uses center point of the arc, can swith to arc pixel locations for future / other center point
        transducer_locs = transducer_locs.transpose((1,0))
        self.transducer_locs = torch.tensor(transducer_locs, dtype=torch.int) 


        # Generate array of coordinates / sensor
        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        xx, yy = np.meshgrid(x, y)
        locations = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        self.sensor_locations = locations
        

    def __len__(self):
        if self.loading_method =='individual':
            return len(self.simulation_paths) # image, tran loc, simu -> 1 sample
        elif self.loading_method == 'group':
            return len(self.image_paths) # image, 8 tran loc, 8 simu -> 1 sample
        else:
            print("can't recognize loading method")
        
    

    def simulation_indv_load(self, index):
        # Process Simulations Individually
        simulations = np.array(Image.open(self.simulation_paths[index]).convert('RGB'))[220:415, 100:710,:]
        simulations = np.transpose(simulations, (2, 0, 1))
        return simulations
    
    def simulation_group_load(self, index):
        #Process all Simulations for a individual image
        simulations = []
        for i in range(index*8,(index+1)*8):
            simulation = np.array(Image.open(self.simulation_paths[i]).convert('RGB'))[220:415, 100:710,:]
            simulation = np.transpose(simulation, (2, 0, 1))
            simulations+=[simulation]
        simulations = np.array(simulations)
        simulations = torch.tensor(simulations, dtype=torch.float) 
        return simulations
    
    def image_load(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')

        if self.image_transforms:
            image_transforms = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = image_transforms(image)
        else:
            image = np.array(image)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
        
        return image

    def eval_locs(self, type = 'sq'):
        #Linear
        if type == 'lin':
            locations = self.sensor_locations # array (x,y)
        #Square
        elif type == 'sq':
            locations = self.sensor_locations.reshape(self.width,self.height,2).transpose(1,0,2) # array of array of (x,y )
        #Cubic
        else:
            ...
        return torch.tensor(locations, dtype=torch.int)

    def __getitem__(self, index):
        if self.loading_method =='individual':
            #Images
            image = self.image_load(index//8)

            #Transducer location
            transducer_locs = self.transducer_locs[index%8]

            #evaludation location (sensor for deepOnet)
            locs = self.eval_locs(type = 'sq')

            #Simulation
            simulations = self.simulation_indv_load(index)


        elif self.loading_method == 'group':
            #Images
            image = self.image_load(index)

            #Transducer location
            transducer_locs = self.transducer_locs

            #evaludation location (sensor for deepOnet)
            loc = self.eval_locs(type = 'sq')
            locs_expanded = loc.unsqueeze(0)
            # Repeat along the new dimension to get shape (8, w, h, 2)
            locs= locs_expanded.repeat(8, 1, 1, 1)

            #Simulation
            simulations = self.simulation_group_load(index)
        else:
            ...

        return image, transducer_locs, locs, simulations
    

"""
# Example Run Case:

images_path, masks_path, simulation_path = get_paths('data')

dataset = TransducerDataset(images_path, simulation_path, loading_method = 'individual')

# Create the DataLoader
bz = 10
data_loader = DataLoader(dataset, batch_size=bz, shuffle=True, num_workers=2)

# Iterating through the DataLoader

i=0
for batch in data_loader:
    images, simulations, transducer_locs = batch
    print(images.shape, simulations.shape, transducer_locs.shape)
    i+=1
    if i >10:
        break
"""