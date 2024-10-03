import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import torchvision.models

###################################64X64#################################################
#######################################Model1#################
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(BaseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(DownConv, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size, padding, stride)
    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels, kernel_size, padding, stride):
        super(UpConv, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_block = BaseConv(in_channels=in_channels + in_channels_skip, out_channels=out_channels,
                                   kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x, x_skip):
        x = self.upconv(x)#x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x
        
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size, padding, stride):
        super(UNet, self).__init__()
        self.act1 = nn.ReLU()
        self.init_conv = BaseConv(in_channels, out_channels, kernel_size, padding, stride)
        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size, padding, stride)
        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size, padding, stride)
        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size, padding, stride)
        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels, kernel_size, padding, stride)
        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels, kernel_size, padding, stride)
        self.up1 = UpConv(2 * out_channels, out_channels, out_channels, kernel_size, padding, stride)
        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)
        #self.out = nn.Linear(out_channels, n_class)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # Decoder
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        x_out = self.out(x_up)
        return x_out

###############################Model2################
import torch.nn.init as init
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") #kernal(4), stride(2), padding(1)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels), #BatchNorm2d
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )
        nn.init.kaiming_normal_(self.conv[0].weight)  # Initialize the Conv2d layer's weights
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self,x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class UNET(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,features=64):
        super().__init__() #kernel, stride, padding = 4,2,1, Input size= 64X
        self.initial_down = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),nn.LeakyReLU(0.2))  # 32*32
        
        ##############################################################################
        ################################## ENCODER ###################################
        ##############################################################################
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=True)    # 16 X 16
        self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout=True)  # 8 X 8
        #self.down3 = Block(features*4, features*8, down=True, act="leaky", use_dropout=True)  # 4 X 4
        #self.down4 = Block(features*8, features*8, down=True, act="leaky", use_dropout=True)  # 2 X 2
        #self.down5 = Block(features*8, features*8, down=True, act="leaky", use_dropout=True)  # 4 X 4
        #self.down6 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False)  # 2 X 2
        ##############################################################################
        ################################# BOTTLENECK #################################
        ##############################################################################
        self.bottleneck = nn.Sequential(nn.Conv2d(features*4,features*4,4,2,1,padding_mode="reflect"), nn.ReLU()) # 1 X 1
        nn.init.kaiming_normal_(self.bottleneck[0].weight)  # Initialize the Conv2d layer's weights
        ##############################################################################
        ################################## DECODER ###################################
        ##############################################################################
        self.up1 = Block(features*4, features*4, down=False, act="lrelu", use_dropout=True) #2X2
        #self.up2 = Block(features*8*2, features*8, down=False, act="lrelu", use_dropout=True) #4
        #self.up3 = Block(features*8*2, features*4, down=False, act="lrelu", use_dropout=True) #8
        self.up4 = Block(features*4*2, features*2, down=False, act="lrelu", use_dropout=True) #16
        #self.up5 = Block(features*2*2, features, down=False, act="relu", use_dropout=False) #32
        #self.up6 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=False) #64
        self.up7 = Block(features*2*2, features, down=False, act="lrelu", use_dropout=True) #128
        self.final_up = nn.Sequential(nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1)) #256 #nn.Tanh()  
        nn.init.kaiming_normal_(self.final_up[0].weight)
        #self.output = nn.Conv2d(out_channels,out_channels,kernel_size=4, stride=2, padding=1), nn.ReLU()
         # Initialize weights using Xavier initialization
        #self.initialize_weights()
        self.elu = nn.ELU() 
    '''
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    '''
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        #d4 = self.down3(d3)
        #d5 = self.down4(d4)
        #d6 = self.down5(d5)
        #d7 = self.down6(d6)
        
        bottleneck = self.bottleneck(d3)
        
        up1 = self.up1(bottleneck)
        #up2 = self.up2(torch.cat([up1, d5], 1))
        #up3 = self.up3(torch.cat([up2, d4], 1))
        up4 = self.up4(torch.cat([up1, d3], 1))
        #up5 = self.up5(torch.cat([up4, d2], 1))
        #up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up4, d2], 1))
        
        return torch.sigmoid(self.final_up(torch.cat([up7, d1],1)))
#self.final_up(torch.cat([up7, d1],1))#self.elu(self.final_up(torch.cat([up7, d1],1)))#torch.sigmoid(self.final_up(torch.cat([up7, d1],1)))