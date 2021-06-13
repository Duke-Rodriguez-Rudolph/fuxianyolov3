from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms





#创建解码盒子,将从网络里出来的特征图进行解码
#进行iou计算与非极大抑制计算
class DecodeBox(nn.Module):
    def __init__(self,anchors,num_classes,img_size):
        super().__init__()
        self.anchors=anchors
        self.num_anchors=len(anchors)
        self.num_classes=num_classes
        self.attrs=5+num_classes
        self.img_size=img_size

    def forward(self,input):
        # batch_size,255,13,13
        # batch_size,255,26,26
        # batch_size,255,52,52
        batch_size=input.size(0)
        input_height=input.size(2)
        input_width=input.size(3)

        stride_h=self.img_size[1]/input_height
        stride_w=self.img_size[0]/input_width

        #将anchors弄成大中小三维度的大小
        scaled_anchors=[(anchor_width/stride_w,anchor_height/stride_h)for anchor_width,anchor_height in self.anchors]

        #将batch_size,255,13,13弄成batch_size,3,13,13,85
        prediction=input.view(batch_size,self.num_anchors,self.attrs,
                              input_height,input_width).permute(0,1,3,4,2).contiguous()

        #attrs:tx,ty,tw,th,conf,......
        x=torch.sigmoid(prediction[...,0])#将最后一维里的东西逐个抽出来#torch.Size([1, 3, 13, 13])
        y=torch.sigmoid(prediction[...,1])#torch.Size([1, 3, 13, 13])
        tw=prediction[...,2]
        th=prediction[...,3]
        conf=torch.sigmoid(prediction[...,4])#torch.Size([1, 3, 13, 13])
        pred_cls=torch.sigmoid(prediction[...,5:])#torch.Size([1, 3, 13, 13, 80])

        FloatTensor=torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor=torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #下面生成网络，制作基础量
         #torch.linspace为生成等差数列，repeat（竖着复制多少，横着复制多少）先制造网格，随后弄出batch*先验框数量的份数
        #用x的shape来reshape，是将batch与先验框数量拆开
        net_x=torch.linspace(0,input_width-1,input_width).repeat(input_height,1).repeat(batch_size*self.num_anchors,1,1).view(x.shape).type(FloatTensor)
        net_y=torch.linspace(0,input_height-1,input_height).repeat(input_width,1).t().repeat(batch_size*self.num_anchors,1,1).view(y.shape).type(FloatTensor)
        #torch.Size([1, 3, 13, 13])
        #torch.Size([1, 3, 13, 13])
        
        #弄成一个二维的Tensor，然后分别将先验框的w与h拆出来，然后弄成与上面网格一样得到格式
        anchor_w=FloatTensor(scaled_anchors).index_select(1,LongTensor([0]))
        anchor_h=FloatTensor(scaled_anchors).index_select(1,LongTensor([1]))
        anchor_w=anchor_w.repeat(batch_size,1).repeat(1,1,input_height*input_width).view(tw.shape)
        anchor_h=anchor_h.repeat(batch_size,1).repeat(1,1,input_height*input_width).view(th.shape)
        #[1.3,13,13]
        #[1.3,13,13]其中3为三个维度对应的anchor
        
        #弄出一个预测盒子，储存新的前四个数据
        pred_boxes=FloatTensor(prediction[...,:4].shape)
        pred_boxes[...,0]=x.data+net_x
        pred_boxes[...,1]=y.data+net_y
        pred_boxes[...,2]=torch.exp(tw.data)*anchor_w
        pred_boxes[...,3]=torch.exp(th.data)*anchor_h

        #将感受野弄成四份，还原原来的大小
        scale=torch.Tensor([stride_w,stride_h]*2).type(FloatTensor)
        output=torch.cat((pred_boxes.view(batch_size,-1,4)*scale,
                          conf.view(batch_size,-1,1),pred_cls.view(batch_size,-1,self.num_classes)),-1)
        #[1,507,4]
        #torch.Size([1, 507, 85])
        return output.data

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):#假设input为416，416，而image为500，500
    #先计算出这个框框在原图所占的比例，然后再×原图大小
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes
'''
def iou(box1,box2):
    b1_x1,b1_y1,b1_x2,b1_y2=box1[:,0],box1[:,1],box1[:,2],box1[:,3]
    b2_x1,b2_y1,b2_x2,b2_y2=box2[:,0],box2[:,1],box2[:,2],box2[:,3]
    inter_rect_x1=torch.max(b1_x1,b2_x1)
    inter_rect_y1=torch.max(b1_y1,b2_y1)
    inter_rect_x2=torch.min(b1_x2,b2_x2)
    inter_rect_y2=torch.min(b1_y2,b2_y2)

    inter_area=torch.clamp(inter_rect_x2-inter_rect_x1+1,min=0)*torch.clamp(inter_rect_y2-inter_rect_y1+1,min=0)

    b1_area=(b1_x2-b1_x1+1)*(b1_y2-b1_y1+1)
    b2_area=(b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)

    iou=inter_area/(b1_area+b2_area-inter_area+1e-16)

    return iou
'''
def non_max_suppression(prediction,num_classes,conf_thres=0.5,nms_thres=0.4):
    #转换为左上角右下角格式
    box_corner=prediction.new(prediction.shape)
    box_corner[:,:,0]=prediction[:,:,0]-prediction[:,:,2]/2
    box_corner[:,:,1]=prediction[:,:,1]-prediction[:,:,3]/2
    box_corner[:,:,2]=prediction[:,:,0]+prediction[:,:,2]/2
    box_corner[:,:,3]=prediction[:,:,1]+prediction[:,:,3]/2
    prediction[:,:,:4]=box_corner[:,:,:4]
    #前四个换成x1,y1,x2,y2
    
    output=[None for i in range(len(prediction))]
    for image_i,image_pred in enumerate(prediction):
        print('image_pred[:,5:5+num_classes]',image_pred[:,5:5+num_classes])
        print(torch.max(image_pred[:,4],-1,keepdim=True))
        class_conf,class_pred=torch.max(image_pred[:,5:5+num_classes],1,keepdim=True)
        print('class_conf:',class_conf)
        conf_mask=(image_pred[:,4]*class_conf[:,0]>=conf_thres).squeeze()

        image_pred=image_pred[conf_mask]
        class_conf=class_conf[conf_mask]
        class_pred=class_pred[conf_mask]
        
        if not image_pred.size(0):
            continue
        detections=torch.cat((image_pred[:,:5],class_conf.float(),class_pred.float()),1)

        unique_labels=detections[:,-1].cpu().unique()
        
        if prediction.is_cuda:
            unique_labels=unique_labels.cuda()
            detections=detections.cuda()

        for c in unique_labels:
            detections_class=detections[detections[:,-1]==c]
            keep=nms(
                detections_class[:,:4],
                detections_class[:,4]*detections_class[:,5],
                nms_thres)
            max_detections=detections_class[keep]

            output[image_i]=max_detection if output[image_i] is None else torch.cat(
                (output[image_i],max_detections))
    return output
    
    
