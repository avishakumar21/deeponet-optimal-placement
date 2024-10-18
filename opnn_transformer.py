import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: [batch_size, embed_dim, H // patch_size, W // patch_size]
        x = x.flatten(2)  # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x


class TransformerBranch(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_dim, dropout, num_patches):
        super(TransformerBranch, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Define transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final MLP to reduce dimensions
        self.fc = nn.Linear(embed_dim, 64)  # Match the output size to 64 as needed
    
    def forward(self, x):
        # Add positional encoding
        x += self.positional_encoding
        x = self.transformer_encoder(x)  # Apply the transformer layers
        
        # Aggregate patches by mean pooling over the patch dimension
        x = x.mean(dim=1)  # Shape: [batch_size, embed_dim]
        
        # Pass through final MLP
        x = self.fc(x)
        return x


class opnn(nn.Module):
    def __init__(self, branch1_dim, branch2_dim, trunk_dim, geometry_dim, output_dim, patch_size=16, embed_dim=256, num_heads=16, num_layers=6, mlp_dim=1024, dropout=0.1):
        super(opnn, self).__init__()

        # Patch embedding for the transformer branch
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, embed_dim=embed_dim)

        # Calculate number of patches
        num_patches = (geometry_dim[0] // patch_size) * (geometry_dim[1] // patch_size)

        # Transformer branch for geometry
        self.transformer_branch = TransformerBranch(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim, dropout=dropout, num_patches=num_patches)

        # Source location branch
        self._branch2 = nn.Sequential(
            nn.Linear(branch2_dim[0], branch2_dim[1]),
            nn.ReLU(),
            nn.Linear(branch2_dim[1], branch2_dim[2]),
            nn.ReLU(),
            nn.Linear(branch2_dim[2], 64)  # Adjust output to 64 features to match y_br
        )

        # Trunk network (adjust output to 64 features)
        self._trunk = nn.Sequential(
            nn.Linear(trunk_dim[0], trunk_dim[1]),
            nn.Tanh(),
            nn.Linear(trunk_dim[1], trunk_dim[2]),
            nn.Tanh(),
            nn.Linear(trunk_dim[2], 64)  # Adjust trunk output to 64 features
        )

    def forward(self, geometry, source_loc, coords):
        # Process geometry image through patch embedding and transformer
        x = self.patch_embedding(geometry)

        y_br1 = self.transformer_branch(x)

        # Process source location through FC network
        y_br2 = self._branch2(source_loc)

        # Combine branch outputs
        y_br = y_br1 * y_br2

        # Process coordinates through trunk network
        y_tr = self._trunk(coords)

        # Perform tensor product over the last dimension of y_br and y_tr
        y_out = torch.einsum("ij,k...j->ik...", y_br, y_tr)
        return y_out
    
    
    def loss(self, geometry, source_loc, coords, target_pressure):
        y_out = self.forward(geometry, source_loc, coords)
        loss = ((y_out - target_pressure) ** 2).mean()
        return loss
    
    # def loss(self, geometry, source_loc, coords, target_pressure):
    #     # Get the predictions from the model
    #     y_out = self.forward(geometry, source_loc, coords)

    #     # Normalize the predictions to the same scale as the targets
    #     y_out = (y_out - y_out.mean()) / (y_out.std() + 1e-8)

    #     # Calculate the loss
    #     loss = ((y_out - target_pressure) ** 2).mean()
    #     return loss


    # def loss(self, geometry, source_loc, coords, target_pressure):
        # y_out = self.forward(geometry, source_loc, coords)
        # y_out = y_out.mean(dim=1)  # New shape: [16, 162, 512]
        # assert y_out.shape == target_pressure.shape, f"Shape mismatch: {y_out.shape} vs {target_pressure.shape}"

        # criterion = nn.SmoothL1Loss()  # or nn.MSELoss()
        # loss = criterion(y_out, target_pressure)
        # l2_reg = sum(param.norm(2) for param in self.parameters())  # L2 regularization
        # return loss + 1e-4 * l2_reg

