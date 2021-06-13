from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

#这里使用darknet网络，以及搭建特征提取金字塔

class yolobody(nn.Module):
    def __init__(self,anchor,num_classes):
        super().__init__()

        self.backbone = darknet53(None)
        #获取每个残差块的输出频道
        out_filters = self.backbone.layers_out_filters
        
        final_out_filter1=len(anchor[0])*(5+num_classes)
        self.last_layer1=self.threefeatures([512,1024],out_filters[-1],final_out_filter1)

        final_out_filter2=len(anchor[1])*(5+num_classes)
        self.last_layer2_conv=self.conv2d(512,256,1)
        self.last_layer2_upsample=nn.Upsample(scale_factor=2,mode='nearest')
        self.last_layer2=self.threefeatures([256,512],out_filters[-2]+256,final_out_filter2)

        final_out_filter3=len(anchor[2])*(5+num_classes)
        self.last_layer3_conv=self.conv2d(256,128,1)
        self.last_layer3_upsample=nn.Upsample(scale_factor=2,mode='nearest')
        self.last_layer3=self.threefeatures([128,256],out_filters[-3]+128,final_out_filter3)

    def forward(self,x):
        def otherway(last_layer,x_in):
            for i,e in enumerate(last_layer):
                x_in=e(x_in)

                if i==4:
                    other_way=x_in
            return x_in,other_way
        
        x3,x2,x1=self.backbone(x)

        out1,other1way=otherway(self.last_layer1,x1)

        x2_in=self.last_layer2_conv(other1way)
        x2_in=self.last_layer2_upsample(x2_in)
        out2,other2way=otherway(self.last_layer2,torch.cat([x2_in,x2],1))

        x3_in=self.last_layer3_conv(other2way)
        x3_in=self.last_layer3_upsample(x3_in)
        out3,_=otherway(self.last_layer3,torch.cat([x3_in,x3],1))

        return out1,out2,out3
        
    def conv2d(self,inplane,outplane,kernel_size):
        pad=(kernel_size-1)//2 if kernel_size else 0
        return nn.Sequential(OrderedDict([
            ('conv',nn.Conv2d(inplane,outplane,kernel_size=kernel_size,stride=1,padding=pad,bias=False)),
            ('bn',nn.BatchNorm2d(outplane)),
            ('relu',nn.LeakyReLU(0.1)),
            ]))

    def threefeatures(self,filters_list,inplane,outplane):
        return nn.ModuleList([
            self.conv2d(inplane,filters_list[0],1),
            self.conv2d(filters_list[0],filters_list[1],3),
            self.conv2d(filters_list[1],filters_list[0],1),
            self.conv2d(filters_list[0],filters_list[1],3),
            self.conv2d(filters_list[1],filters_list[0],1),
            self.conv2d(filters_list[0],filters_list[1],3),
            nn.Conv2d(filters_list[1],outplane,kernel_size=1,stride=1,padding=0,bias=True)
            ])


