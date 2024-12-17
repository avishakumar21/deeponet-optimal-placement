import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class opnn_resnet(nn.Module):
    def __init__(self, branch2_dim, trunk_dim, geometry_dim):
        super(opnn_resnet, self).__init__()
        # Load a pretrained ResNet model
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Additional layers for upsampling and reducing channels
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1 output channel for regression
        
    def forward(self, geometry, source_loc, coords):
        x = self.backbone(geometry)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, size=(162, 512), mode='bilinear', align_corners=False)  # Ensure output is 162x512
        x = self.output_conv(x)
        x = x.squeeze(1)
        return x
    
    def loss(self, geometry, source_loc, coords, target_pressure):
        y_out = self.forward(geometry, source_loc, coords)
        numerator = torch.norm(y_out - target_pressure, p=2)
        denominator = torch.norm(target_pressure, p=2)  # Avoid division by zero
        loss = (numerator / denominator) ** 2
        return loss

class opnn_cnn(nn.Module):
    def __init__(self, branch2_dim, trunk_dim, geometry_dim):
        super(opnn_cnn, self).__init__()
        # Define the CNN for the geometry image (Branch 1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Upsampling layers to achieve 162x512 output
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(size=(162, 512), mode='bilinear', align_corners=False)

        # Final output convolution to produce 1-channel output for regression
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1 output channel for regression
        
    def forward(self, geometry, source_loc, coords):
        # Pass through the convolutional layers
        x = F.relu(self.conv1(geometry))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downscales by 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Upsampling layers to reach the target size
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)

        # Final output layer for regression
        x = self.output_conv(x)
        
        # Remove the singleton channel dimension to output (162, 512)
        x = x.squeeze(1)  # Output shape: (batch_size, 162, 512)
        return x
    
    def loss(self, geometry, source_loc, coords, target_pressure):
        y_out = self.forward(geometry, source_loc, coords)
        numerator = torch.norm(y_out - target_pressure, p=2)
        denominator = torch.norm(target_pressure, p=2)  # Avoid division by zero
        loss = (numerator / denominator) ** 2
        return loss