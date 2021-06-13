import math
from collections import OrderedDict
import torch.nn as nn
import torch


#残差块单独做成一个类，darknet53也做成一个类
class Residualblock(nn.Module):
    def __init__(self,planes):
        super().__init__()
        self.conv1=nn.Conv2d(planes[1],planes[0],kernel_size=1,
                             stride=1,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(planes[0])
        self.relu1=nn.LeakyReLU(0.1)

        self.conv2=nn.Conv2d(planes[0],planes[1],kernel_size=3,
                             stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes[1])
        self.relu2=nn.LeakyReLU(0.1)

    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu1(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu2(out)

        out+=residual
        return out

class darknet(nn.Module):
    def __init__(self,nums):
        super().__init__()
        #darknet第一步卷积
        self.conv1=nn.Conv2d(3,32,kernel_size=3,
                             stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(32)
        self.relu1=nn.LeakyReLU(0.1)

        #残差块疯狂堆叠
        self.layer1=self.reResi([32,64],nums[0])
        self.layer2=self.reResi([64,128],nums[1])
        self.layer3=self.reResi([128,256],nums[2])
        self.layer4=self.reResi([256,512],nums[3])
        self.layer5=self.reResi([512,1024],nums[4])

        self.layers_out_filters=[64,128,256,512,1024]

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def reResi(self,planes,num):
        layers=[]
        #第一步卷积
        layers.append(('RB_conv',nn.Conv2d(planes[0],planes[1],kernel_size=3,
                                           stride=2,padding=1,bias=False)))
        layers.append(('RB_bn',nn.BatchNorm2d(planes[1])))
        layers.append(('RB_relu',nn.LeakyReLU(0.1)))

        #进行残差块的堆叠
        for i in range(0,num):
            layers.append(('residual_{}'.format(i),Residualblock(planes)))
        return nn.Sequential(OrderedDict(layers))
            

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.layer1(x)
        x=self.layer2(x)
        out1=self.layer3(x)
        out2=self.layer4(out1)
        out3=self.layer5(out2)

        return out1,out2,out3

def darknet53(pretrained,**kwargs):
    model=darknet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained,str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
        
    
