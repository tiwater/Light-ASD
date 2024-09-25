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
            x = x.view(B * D, C, H, W)  # (N * D, C, H, W)
            x = self.maxpool2d1(x)
            _, C, H, W = x.shape
            x = x.view(B, D, C, H, W)  # (N, D, C, H, W)
            x = x.transpose(1, 2)  # (N, D, C, H, W) -> (N, C, D, H, W)

            # Apply 2D pooling along D and one of H or W (since kernel_size2d2 aims at depth)
            x = x.transpose(2, 3)  # (N, C, D, H, W) -> (N, C, H, D, W)
            B, C, H, D, W = x.shape
            x = x.view(B * C, H, D, W)  # (N * C, H, D, W)
            x = self.maxpool2d2(x)
            H, D, W = x.shape[1:]
            x = x.view(B, C, H, D, W)  # (N, C, H, D, W)
            x = x.transpose(2, 3)  # (N, C, H, D, W) -> (N, C, D, H, W)
        else:
            x = self.maxpool2d1(x)
        return x
    
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv3d, self).__init__()

        # Ensure stride and padding are always treated as tuples of length 3
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_size2d_1 = kernel_size[1:]
        stride2d_1 = stride[1:]
        padding2d_1 = padding[1:]

        kernel_size2d_2 = (kernel_size[0], 1)
        stride2d_2 = (stride[0], 1)
        padding2d_2 = (padding[0], 0)

        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size2d_1, stride=stride2d_1, padding=padding2d_1, bias=bias)
        self.conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size2d_2, stride=stride2d_2, padding=padding2d_2, bias=bias)

    
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.transpose(1, 2).reshape(B * D, C, H, W)  # (N, C, D, H, W) -> (N, D, C, H, W)
        x = self.conv2d_1(x)

        _, C, H, W = x.shape
        x = x.reshape(B, D, C, H, W)  # (N, D, out_channels, H, W)
        x = x.transpose(1, 2)  # (N, out_channels, D, H, W)

        B, C, D, H, W = x.shape
        x = x.reshape(B * H, C, D, W)  # (N * H, out_channels, D, W)
        x = self.conv2d_2(x)

        _, C, D, W = x.shape
        x = x.reshape(B, H, C, D, W)  # (N, H, out_channels, D, W)
        x = x.transpose(1, 2).transpose(2, 3)  # (N, out_channels, D, H, W)
        
        return x
    
class BatchNorm3d(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=0.001):
        super(BatchNorm3d, self).__init__()
        self.bn2d = nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.transpose(1, 2).reshape(B * D, C, H, W)
        x = self.bn2d(x)
        x = x.view(B, D, C, H, W).transpose(1, 2)
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

        stride = (1, 2, 2) if is_down else (1, 1, 1)
        padding_3 = (0, 1, 1)
        padding_5 = (0, 2, 2)

        self.s_3 = Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=stride, padding=padding_3, bias=False)
        self.bn_s_3 = BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn_t_3 = BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.s_5 = Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), stride=stride, padding=padding_5, bias=False)
        self.bn_s_5 = BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = Conv3d(out_channels, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False)
        self.bn_t_5 = BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.last = Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn_last = BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = x.float()
        x_3 = self.relu(self.bn_s_3(self.s_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_s_5(self.s_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5
        x = self.relu(self.bn_last(self.last(x)))

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