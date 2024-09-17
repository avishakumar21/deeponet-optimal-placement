import torch
import torch.nn as nn
import torch.nn.functional as F

class opnn(nn.Module):
    def __init__(self, branch1_dim, branch2_dim, trunk_dim, geometry_dim, output_dim):
        super(opnn, self).__init__()
        
        # Define the CNN for the geometry image (Branch 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (geometry_dim // 8) * (geometry_dim // 8), branch1_dim[-1])  # Adjust dimensions after pooling

        # Define the small fully connected network for source location (Branch 2)
        self._branch2 = nn.Sequential(
            nn.Linear(branch2_dim[0], branch2_dim[1]),
            nn.ReLU(),
            nn.Linear(branch2_dim[1], branch2_dim[2]),
            nn.ReLU(),
            nn.Linear(branch2_dim[2], branch2_dim[-1])
        )

        # Define the trunk network (coordinates)
        self._trunk = nn.Sequential(
            nn.Linear(trunk_dim[0], trunk_dim[1]),
            nn.Tanh(),
            nn.Linear(trunk_dim[1], trunk_dim[2]),
            nn.Tanh(),
            nn.Linear(trunk_dim[2], trunk_dim[-1])
        )
    
    def forward(self, geometry, source_loc, coords):
        # Process geometry image through CNN
        x = F.relu(self.conv1(geometry))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        y_br1 = F.relu(self.fc1(x))
        
        # Process source location through FC network
        y_br2 = self._branch2(source_loc)
        
        # Combine branch outputs
        y_br = y_br1 * y_br2

        # Process coordinates through trunk network
        y_tr = self._trunk(coords)

        # Perform tensor product over the last dimension of y_br and y_tr
        y_out = torch.einsum("ij,kj->ik", y_br, y_tr)
        return y_out
    
    def loss(self, geometry, source_loc, coords, target_pressure):
        y_out = self.forward(geometry, source_loc, coords)
        loss = ((y_out - target_pressure) ** 2).mean()
        return loss
