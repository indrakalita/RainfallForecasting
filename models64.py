import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import torchvision.models

###################################64X64#################################################

###########################Model4########################
class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features, mean, std):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.mean = mean
        self.std = std

    def forward(self, x):
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        x = (x - mean) / (std + 1e-5)
        return x
class SpatialDropout(nn.Module):  # Define SpatialDropout as a nested class
      def __init__(self, drop_rate):
        super(SpatialDropout, self).__init__()
        self.drop_rate = drop_rate

      def forward(self, x):
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.drop_rate))
        return x * mask
class BaseG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dropout=False, norm='custom', activation='relu', mean=0, std=1):
        super(BaseG, self).__init__()
        self.act = nn.ReLU() if activation == 'relu' else (nn.LeakyReLU(negative_slope=0.2) if activation == 'leaky' else nn.Identity() if activation == 'linear' else None) #ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
        #self.norm1 = nn.BatchNorm2d(out_channels) if norm =='batch' else (nn.InstanceNorm2d(out_channels) if norm == 'instance' else None)
        #self.norm2 = nn.BatchNorm2d(out_channels) if norm =='batch' else (nn.InstanceNorm2d(out_channels) if norm == 'instance' else None)
        self.norm1 = CustomBatchNorm2d(out_channels, mean, std) if norm == 'custom' else (nn.BatchNorm2d(out_channels) if norm == 'batch' else (nn.InstanceNorm2d(out_channels) if norm == 'instance' else None))
        self.norm2 = CustomBatchNorm2d(out_channels, mean, std) if norm == 'custom' else (nn.BatchNorm2d(out_channels) if norm == 'batch' else (nn.InstanceNorm2d(out_channels) if norm == 'instance' else None))
        #self.dropout1 = SpatialDropout(0.5) if dropout else None
        #self.dropout2 = SpatialDropout(0.5) if dropout else None
        self.dropout1 = nn.Dropout(0.3) if dropout else None
        self.dropout2 = nn.Dropout(0.3) if dropout else None
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.norm1(x) if self.norm1 is not None else x
        x = self.dropout1(x) if self.dropout1 else x
        x = self.act(self.conv2(x))
        x = self.norm2(x) if self.norm1 is not None else x
        x = self.dropout2(x) if self.dropout2 else x
        return x

class DownG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dropout, norm, activation, mean, std):
        super(DownG, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseG(in_channels, out_channels, kernel_size, padding, stride, dropout, norm, activation, mean=mean, std=std)
    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x

class UpG(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels, kernel_size, padding, stride, dropout, norm, activation, mean, std):
        super(UpG, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_block = BaseG(in_channels=in_channels + in_channels_skip, out_channels=out_channels,
                                   kernel_size=kernel_size, padding=padding, stride=stride, dropout=dropout, norm=norm, activation=activation, mean=mean, std=std)

    def forward(self, x, x_skip):
        #x = self.conv_trans1(x)
        x = self.upconv(x)
        #print(x.shape)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x
        
class GNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size, padding, stride, mean, std):
        super(GNet, self).__init__()
        self.norm = 'instance'
        self.act = 'relu'
        self.act1 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.2)
        self.init_conv = BaseG(in_channels, out_channels, kernel_size, padding, stride,
                                   dropout=True, norm=self.norm, activation=self.act, mean=mean, std=std) #64-->64
        self.down1 = DownG(out_channels, 2 * out_channels, kernel_size, padding, stride,
                              dropout=True, norm=self.norm, activation=self.act, mean=mean, std=std) #64-->32
        self.down2 = DownG(2 * out_channels, 4 * out_channels, kernel_size, padding, stride,
                             dropout=True, norm=self.norm, activation=self.act, mean=mean, std=std) #32-->16
        self.down3 = DownG(4 * out_channels, 8 * out_channels, kernel_size, padding, stride,
                             dropout=True, norm=self.norm, activation=self.act, mean=mean, std=std) #16-->8
        
        
        self.up3 = UpG(8 * out_channels, 4 * out_channels, 4 * out_channels, kernel_size, padding, stride,
                         dropout=True, norm=self.norm, activation=self.act, mean=mean, std=std) #8-->16
        self.up2 = UpG(4 * out_channels, 2 * out_channels, 2 * out_channels, kernel_size, padding, stride,
                         dropout=True, norm=self.norm, activation=self.act, mean=mean, std=std) #16-->32
        self.up1 = UpG(2 * out_channels, out_channels, out_channels, kernel_size, padding, stride,
                         dropout=True, norm=self.norm, activation=self.act, mean=mean, std=std) #32-->64
        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride) #64-->64

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
        x_out = self.out(x_up)#self.act1(self.out(x_up))
        return x_out

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

#################Model3###################
# A pre-trained based UNET
import torchvision.models
def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, padding=padding),nn.ReLU(inplace=True))
    
class ResNetUNet(nn.Module):
  def __init__(self, n_class):
    super().__init__()
    self.base_model = torchvision.models.resnet18(pretrained=True)
    self.base_layers = list(self.base_model.children())

    self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer0_1x1 = convrelu(64, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer1_1x1 = convrelu(64, 64, 1, 0)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer2_1x1 = convrelu(128, 128, 1, 0)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer3_1x1 = convrelu(256, 256, 1, 0)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.layer4_1x1 = convrelu(512, 512, 1, 0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
    self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
    self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
    self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
    self.conv_original_size0 = convrelu(3, 64, 3, 1)
    self.conv_original_size1 = convrelu(64, 64, 3, 1)
    self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

    self.conv_last = nn.Conv2d(64, n_class, 1)  
    #self.conv_last1 = nn.Linear(64, n_class)

  def forward(self, input):
    x_original = self.conv_original_size0(input)
    x_original = self.conv_original_size1(x_original)
    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)
    layer4 = self.layer4_1x1(layer4)
    x = self.upsample(layer4)
    layer3 = self.layer3_1x1(layer3)
    x = torch.cat([x, layer3], dim=1)
    x = self.conv_up3(x)
    x = self.upsample(x)
    layer2 = self.layer2_1x1(layer2)
    x = torch.cat([x, layer2], dim=1)
    x = self.conv_up2(x)
    x = self.upsample(x)
    layer1 = self.layer1_1x1(layer1)
    x = torch.cat([x, layer1], dim=1)
    x = self.conv_up1(x)
    x = self.upsample(x)
    layer0 = self.layer0_1x1(layer0)
    x = torch.cat([x, layer0], dim=1)
    x = self.conv_up0(x)
    x = self.upsample(x)
    x = torch.cat([x, x_original], dim=1)
    x = self.conv_original_size2(x)
    out = self.conv_last(x)
    return out