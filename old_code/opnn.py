"""
Fast training version DeepONet
Only works for fix grid training data
"""
import torch
import torch.nn as nn
from support_models import CNNModel

class opnn(nn.Module):
    def __init__(self, branch1_dim, branch2_dim, trunk_dim):
        super(opnn, self).__init__()
        # self.branch_dim = branch_dim
        # self.trunk_dim = trunk_dim
        self.z_dim = trunk_dim[-1]

        ## build branch net
        modules = []
        # for i, h_dim in enumerate(branch1_dim):
        #     if i == 0:
        #         in_channels = h_dim
        #     else:
        #         modules.append(nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.Tanh()
        #             )
        #         )
        #         in_channels = h_dim
        # self._branch1 = nn.Sequential(*modules)

        # Branch 1: CNN
        self._branch1 = CNNModel()

        # modules = []
        # for i, h_dim in enumerate(branch2_dim):
        #     if i == 0:
        #         in_channels = h_dim
        #     else:
        #         modules.append(nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.Tanh()
        #             )
        #         )
        #         in_channels = h_dim
        # self._branch2 = nn.Sequential(*modules)
        self._branch2 = nn.Linear(in_features=2, out_features=512)

        ## build trunk net
        # modules = []
        # for i, h_dim in enumerate(trunk_dim):
        #     if i == 0:
        #         in_channels = h_dim
        #     else:
        #         modules.append(nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.Tanh()
        #             )
        #         )
        #         in_channels = h_dim
        # self._trunk = nn.Sequential(*modules)
        self._trunk = nn.Linear(in_features=195 * 610 * 2, out_features=512) #TODO: How to Handle Location (sensor coordinates) Data
        # self._out_layer = nn.Linear(self.z_dim, 1, bias = True)
        self._out_layer = nn.Linear(in_features = 512 * 2, out_features = 162 * 512)

    def forward(self, f, f_bc, x):
        """
        f: M*dim_f
        x: N*dim_x
        y_br: M*dim_h
        y_tr: N*dim_h
        y_out: y_br(ij) tensorprod y_tr(kj)
        """
        bz, h, w, _ = x.shape  # Extract batch size, height, and width
        
        #branch input
        #print(f.shape, f_bc.shape)
        y_br1 = self._branch1(f)
        y_br2 = self._branch2(f_bc)
        y_br = y_br1*y_br2
        #print('y_br shape', y_br.shape)

        #Trunck input
        # y_tr = self._trunk(x)
        bz, h, w, _ = x.shape  # Extract batch size, height, and width
        x_flat = x.view(bz, -1) # Flatten to [batchsize, num_points, 2]
        x_flat = x_flat.to(torch.float32) # TODO: Locate in data loader to convert into FLOAT
        y_tr = self._trunk(x_flat)
        #print('y_tr shape', y_tr.shape)

        # y_out = torch.einsum("ij,kj->ik", y_br, y_tr)
        # y_out = torch.einsum("bi,bji->bj", y_br, y_tr)
        y_out = torch.cat((y_br, y_tr), dim=1) #TODO: Change into dot product instead of concat
        y_out = self._out_layer(y_out)

        #print('y_out shape', y_out.shape)
        y_out = y_out.view(bz, 162, 512)
        #print('y_out shape', y_out.shape)


        return y_out
    
    def loss(self, f, f_bc, x, y):
        y_out = self.forward(f, f_bc, x)
        loss = ((y_out - y)**2).mean()
        return loss
    