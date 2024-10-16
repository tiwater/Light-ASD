import torch
import torch.nn as nn

# The version supports rknn by rewriting 3d operation as 2d operation
class MaxPool_3D(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool_3D, self).__init__()
        
        kernel_size2d1 = kernel_size[-2:]
        stride2d1 = stride[-2:]
        padding2d1 = padding[-2:]

        kernel_size2d2 = (kernel_size[0], kernel_size[0])
        stride2d2 = (stride[0], stride[0])
        padding2d2 = (padding[0], padding[0])

        self.maxpool2d1 = nn.MaxPool2d(kernel_size=kernel_size2d1, stride=stride2d1, padding=padding2d1)
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=kernel_size2d2, stride=stride2d2, padding=padding2d2)

    def forward(self, x):
        if x.dim() == 5:
            x = x.transpose(1, 2)  # (N, C, D, H, W) -> (N, D, C, H, W)
            B, D, C, H, W = x.shape
            x = x.contiguous().view(B * D, C, H, W)  # (N * D, C, H, W)
            x = self.maxpool2d1(x)
            _, C, H, W = x.shape
            x = x.contiguous().view(B, D, C, H, W)  # (N, D, C, H, W)
            x = x.transpose(1, 2)  # (N, D, C, H, W) -> (N, C, D, H, W)

            # Apply 2D pooling along D and one of H or W (since kernel_size2d2 aims at depth)
            x = x.transpose(2, 3)  # (N, C, D, H, W) -> (N, C, H, D, W)
            B, C, H, D, W = x.shape
            x = x.contiguous().view(B * C, H, D, W)  # (N * C, H, D, W)
            x = self.maxpool2d2(x)
            H, D, W = x.shape[1:]
            x = x.contiguous().view(B, C, H, D, W)  # (N, C, H, D, W)
            x = x.transpose(2, 3)  # (N, C, H, D, W) -> (N, C, D, H, W)
        else:
            x = self.maxpool2d1(x)
        return x  

class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()

        self.m_3 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 1), padding = (1, 0), bias = False)
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_3 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 3), padding = (0, 1), bias = False)
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.m_5 = nn.Conv2d(in_channels, out_channels, kernel_size = (5, 1), padding = (2, 0), bias = False)
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_5 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 5), padding = (0, 2), bias = False)
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.last = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 1), padding = (0, 0), bias = False)
        self.bn_last = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):
        x = x.float()
        x_3 = self.relu(self.bn_m_3(self.m_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_m_5(self.m_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5
        x = self.relu(self.bn_last(self.last(x)))

        return x

class Visual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=False):
        super(Visual_Block, self).__init__()

        self.relu = nn.ReLU()

        spatial_stride = (2, 2) if is_down else (1, 1)

        # 3D Conv(1x3x3) equivalent: Split into 2D for spatial and 1D for temporal
        self.s_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=spatial_stride, padding=1, bias=False)
        self.bn_s_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_t_3 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)

        # 3D Conv(1x5x5) equivalent: Split into 2D for spatial and 1D for temporal
        self.s_5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=spatial_stride, padding=2, bias=False)
        self.bn_s_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn_t_5 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)

        # Final adjustment layer
        self.last = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        # Initial shape [N, C, T, H, W]
        n, c, t, h, w = x.size()

        # Path with 3x3 spatial kernel
        x_3 = x.permute(0, 2, 1, 3, 4)  # [N, T, C, H, W]
        x_3 = x_3.reshape(n * t, c, h, w)  # Reshape to [N*T, C, H, W]
        x_3 = self.relu(self.bn_s_3(self.s_3(x_3)))  # Apply 2D conv
        _, out_c, new_h, new_w = x_3.size()

        x_3 = x_3.view(n, t, out_c, new_h, new_w).permute(0, 2, 3, 4, 1)  # Reorder to [N, O, H, W, T]
        x_3 = x_3.reshape(n * out_c * new_h * new_w, t)  # [N*O*H*W, T]
        x_3 = x_3.view(-1, out_c, t)  # Viewing as required for 1D conv [N*O*H*W, Out_C, T]

        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))  # Apply 1D conv across temporal

        x_3 = x_3.view(n, out_c, new_h, new_w, t).permute(0, 1, 4, 2, 3)  # Return back to [N, C, T, H, W]

        # Path with 5x5 spatial kernel
        x_5 = x.permute(0, 2, 1, 3, 4)  # [N, T, C, H, W]
        x_5 = x_5.reshape(n * t, c, h, w)  # Reshape to [N*T, C, H, W]
        x_5 = self.relu(self.bn_s_5(self.s_5(x_5)))  # Apply 2D conv
        x_5 = x_5.view(n, t, out_c, new_h, new_w).permute(0, 2, 3, 4, 1)  # Reorder to [N, O, H, W, T]
        x_5 = x_5.reshape(n * out_c * new_h * new_w, t)  # [N*O*H*W, T]
        x_5 = x_5.view(-1, out_c, t)  # Viewing as required for 1D conv [N*O*H*W, Out_C, T]

        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))  # Apply 1D conv across temporal

        x_5 = x_5.view(n, out_c, new_h, new_w, t).permute(0, 1, 4, 2, 3)  # Return back to [N, C, T, H, W]

        # Element-wise addition of outputs from both paths
        x = x_3 + x_5

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Change to [N, T, C, H, W] for final spatial operation
        x = x.view(n * t, out_c, new_h, new_w)  # Reshape to [N*T, C, H, W]

        # Final 1x1 conv layer
        x = self.relu(self.bn_last(self.last(x)))  # Apply last 2D conv
        x = x.view(n, t, out_c, new_h, new_w).permute(0, 2, 1, 3, 4)  # Return back to [N, C, T, H, W]
        
        return x
    
class visual_encoder(nn.Module):
    def __init__(self):
        super(visual_encoder, self).__init__()

        self.block1 = Visual_Block(1, 32, is_down = True)
        self.pool1 = MaxPool_3D(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))

        self.block2 = Visual_Block(32, 64)
        self.pool2 = MaxPool_3D(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        
        self.block3 = Visual_Block(64, 128)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()     

    def forward(self, x):

        x = x.float()
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = x.transpose(1,2)
        B, T, C, W, H = x.shape  
        x = x.view(B*T, C, W, H)

        x = self.maxpool(x)

        x = x.view(B, T, C)  
        
        return x

    def __init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    def __init__(self):
        super(audio_encoder, self).__init__()
        
        self.block1 = Audio_Block(1, 32)
        self.pool1 = MaxPool_3D(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))

        self.block2 = Audio_Block(32, 64)
        self.pool2 = MaxPool_3D(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))
        
        self.block3 = Audio_Block(64, 128)

        self.__init_weight()
            
    def forward(self, x):

        x = x.float()
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim = 2, keepdim = True)
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)  # Replace squeeze with view
        
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()