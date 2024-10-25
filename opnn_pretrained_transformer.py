import torch
import torch.nn as nn
import timm  

class opnn(nn.Module):
    def __init__(self, branch2_dim, trunk_dim, pretrained_model='vit_base_patch16_224'):
        super(opnn, self).__init__()

        self.transformer_branch = timm.create_model(pretrained_model, pretrained=True)
        self.transformer_branch.head = nn.Linear(self.transformer_branch.head.in_features, 64)  # Adjust output to 64

        # Source location branch (fully connected network)
        self._branch2 = nn.Sequential(
            nn.Linear(branch2_dim[0], branch2_dim[1]),
            nn.ReLU(),
            nn.Linear(branch2_dim[1], branch2_dim[2]),
            nn.ReLU(),
            nn.Linear(branch2_dim[2], branch2_dim[3])  # Output size: 64
        )

        # Trunk network (fully connected network)
        self._trunk = nn.Sequential(
            nn.Linear(trunk_dim[0], trunk_dim[1]),
            nn.Tanh(),
            nn.Linear(trunk_dim[1], trunk_dim[2]),
            nn.Tanh(),
            nn.Linear(trunk_dim[2], branch2_dim[3])  # Output size: 64
        )

    def forward(self, geometry, source_loc, coords):
        # Process geometry image through the pretrained transformer branch
        y_br1 = self.transformer_branch(geometry)

        # Process source location through FC network
        y_br2 = self._branch2(source_loc)

        # Combine branch outputs
        y_br = y_br1 * y_br2

        # Process coordinates through trunk network
        y_tr = self._trunk(coords)

        # Perform tensor product over the last dimension of y_br and y_tr
        y_out = torch.einsum("bf,bhwf->bhw", y_br, y_tr)
        return y_out

    def loss(self, geometry, source_loc, coords, target_pressure):
        y_out = self.forward(geometry, source_loc, coords)
        loss = ((y_out - target_pressure) ** 2).mean()
        return loss
