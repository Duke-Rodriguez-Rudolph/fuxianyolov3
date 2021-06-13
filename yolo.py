import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.yolo3 import yolobody
from utils.utils import (DecodeBox, letterbox_image, non_max_suppression,
                         yolo_correct_boxes)

#创建yolo类
class yolo(object):
    _defaults={
        'model_path':'model_data/yolo_weights.pth',
        'anchors_path':'model_data/yolo_anchors.txt',
        'classes_path':'model_data/coco_classes.txt',
        'model_image_size':(416,416,3),
        'confidence':0.5,
        'iou':0.3,
        'cuda':True,
        'letterbox_image':False,}

    @classmethod
    def get_defaults(cls,n):
        if n in cls._defaults:
            return cls._defaults
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self,**kwargs):
        self.__dict__.update(self._defaults)
        self.classes_names=self._get_class()
        self.anchors=self._get_anchors()
        self.generate()

    def _get_class(self):
        classes_path=os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names=f.readlines()
        class_names=[c.strip() for c in class_names]
        return class_names#一个装有分类的列表

    def _get_anchors(self):
        anchors_path=os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors=f.readline()
        anchors=[float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1,3,2])[::-1,:,:]
        #获得三个大小的先验框，用array装,[[[],[],[]],[[],[],[]],[[],[],[]]]

    def generate(self):
        self.num_classes=len(self.classes_names)
        self.net=yolobody(self.anchors,self.num_classes)#使用nets.yolo3中的yolobody，输入先验框信息以及分类数量
        print('加载权重中……')
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict=torch.load(self.model_path,map_location=device)
        self.net.load_state_dict(state_dict)
        self.net=self.net.eval()

        if self.cuda:
            self.net=nn.DataParallel(self.net)
            self.net=self.net.cuda()

        self.yolo_decodes=[]
         #[[116.  90.]
        #[156. 198.]
        #[373. 326.]]
        #每个self.anchors[i]都是一个二维数组，用array装的
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i],self.num_classes,(self.model_image_size[1],self.model_image_size[0])))

        print('{} 模型, 先验框, 和类别已加载.'.format(self.model_path))
          # 画框设置不同的颜色（为每个类设置一种颜色）
        #弄点颜色
        hsv_tuples=[(x/len(self.classes_names),1.,1.)#[(0.0, 1.0, 1.0), (0.0125, 1.0, 1.0), (0.025, 1.0, 1.0),……]
                    for x in range(len(self.classes_names))]
        self.colors=list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
        self.colors=list(map(lambda x:(int(x[0]*255),int(x[1]*255),int(x[2]*2255)),self.colors))
        
    def detect_image(self,image):
        image=image.convert('RGB')
        image_shape=np.array(np.shape(image)[0:2])
        #[1330,1330]

        if self.letterbox_image:
            crop_img=np.array(letterbox_image(image,(self.model_image_size[1],self,model_image_size[0])))
        else:
            crop_img=image.resize((self.model_image_size[1],self.model_image_size[0]),Image.BICUBIC)
            #[416,416]
        photo=np.array(crop_img,dtype=np.float32)/255.0#归一化
        photo=np.transpose(photo,(2,0,1))#三通道分离
        images=[photo]#增加batch_size维度[array[[[]]]]

        with torch.no_grad():
            images=torch.from_numpy(np.asarray(images))
            if self.cuda:
                images=images.cuda()

            outputs=self.net(images)#应该能输出大中小三个特征图
            output_list=[]
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
                #把三个特征图丢进解码器后的三个输出（预测框）输入输出列表 

            output=torch.cat(output_list,1)
            batch_detections=non_max_suppression(output,self.num_classes,conf_thres=self.confidence,nms_thres=self.iou)
            #一个列表中装一个tensor[(12,7)]
            try:
                batch_detections=batch_detections[0].cpu().numpy()
            except:
                return image

            top_index=batch_detections[:,4]*batch_detections[:,5]>self.confidence
            #[0.99966156 0.9995964  0.9922506  0.98432237 0.97268254 0.9448944 0.89083326 0.76416326 0.99891233 0.9897372  0.93305784 0.837368  ]
            #[0.99997544 0.9999932  0.99994576 0.99991274 0.9999294  0.99978286 0.9999043  0.9974111  0.9998116  0.99549025 0.9888509  0.98151267]
            # [ True  True  True  True  True  True  True  True  True  True  True  True]
            top_conf=batch_detections[top_index,4]*batch_detections[top_index,5]
            #[0.999637   0.9995896  0.9921968  0.9842365  0.9726139  0.9446892 0.89074796 0.7621849  0.9987241  0.9852737  0.9226551  0.8218873 ]
            top_label=np.array(batch_detections[top_index,-1],np.int32)
             #[0. 0. 0. 0. 0. 0. 0. 0. 1. 2. 2. 2.]
            #将上面筛选失败的，全部去掉
            top_bboxes=np.array(batch_detections[top_index,:4])
            top_xmin,top_ymin,top_xmax,top_ymax=np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            if self.letterbox_image:
                boxes=yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
            else:
                top_xmin=top_xmin/self.model_image_size[1]*image_shape[1]
                top_ymin=top_ymin/self.model_image_size[0]*image_shape[0]
                top_xmax=top_xmax/self.model_image_size[1]*image_shape[1]
                top_ymax=top_ymax/self.model_image_size[0]*image_shape[0]
                boxes=np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax],axis=-1)

        font=ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2*np.shape(image)[1]+0.5).astype('int32'))

        thickness=max((np.shape(image)[0]+np.shape(image)[1])//self.model_image_size[0],1)

        for i,c in enumerate(top_label):
            predicted_class=self.class_names[c]
            score=top_conf[i]

            top.left,bottom,right=boxes[i]
            top=top-5
            left=left-5
            bottom=bottom+5
            right=right+5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
            
