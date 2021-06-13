import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
#数据集的加载与处理

class yolodataset(Dataset):
    def __init__(self,train_lines,image_size,is_train):
        super().__init__()
        self.train_lines=train_lines
        self.train_batches=len(train_lines)
        self.image_size=image_size
        self.is_train=is_train

    def __len__(self):
        return self.train_batches
    
    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a)+a
    
    def get_random_data(self,line,input_size,jitter=0.3,hue=0.1,sat=1.5,val=1.5,random=True):
        #负责打乱以及各种折磨训练集的方法
        #读取图片以及标签内容
        line=line.split()
        img=Image.open(line[0])
        img_w,img_h=img.size
        input_h,input_w=input_size
        box=np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        
        if not random:
            #加灰条操作
            scale=min(input_w/img_w,input_h/img_h)
            new_w=int(input_w*scale)
            new_h=int(input_h*scale)
            dx=(input_w-new_w)//2
            dy=(input_h-new_h)//2
            img=img.resize((new_w,new_h),Image.BICUBIC)
            new_img=Image.new('RGB',(input_w,input_h),(128,128,128))
            new_img.paste(img,(dx,dy))
            image_data=np.array(new_img,np.float32)
            #将标签中的坐标调整为中心xy与wh
            box_data=np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:,[0,2]]=box[:,[0,2]]*new_w/input_w+dx
                box[:,[1,3]]=box[:,[1,3]]*new_h/input_h+dy
                box[:,0:2][box[:,0:2]<0]=0
                box[:,2][box[:,2]>input_w]=input_w
                box[:,3][box[:,3]>input_h]=input_h
                box_w=box[:,2]-box[:,0]
                box_h=box[:,3]-box[:,1]
                box=box[np.logical_and(box_w>1,box_h)>1]
                box_data=np.zeros((len(box),5))
                box_data[:len(box)]=box
            return image_data,box_data
        
        #调整大小
        new_ar=input_w/input_h*self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            new_h = int(scale * input_h)
            new_w = int(new_h * new_ar)
        else:
            new_w = int(scale * input_w)
            new_h = int(new_w / new_ar)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        dx = int(self.rand(0, input_w - new_w))
        dy = int(self.rand(0, input_h - new_h))
        new_img = Image.new('RGB', (input_w, input_h), (128, 128, 128))
        new_img.paste(img, (dx, dy))
        img = new_img

        # 是否翻转图片
        flip = self.rand() < .5#随机判断是否要翻转
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(img,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        box_data=np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:,[0,2]]=box[:,[0,2]]*new_w/input_w+dx
            box[:,[1,3]]=box[:,[1,3]]*new_h/input_h+dy
            box[:,0:2][box[:,0:2]<0]=0
            box[:,2][box[:,2]>input_w]=input_w
            box[:,3][box[:,3]>input_h]=input_h
            box_w=box[:,2]-box[:,0]
            box_h=box[:,3]-box[:,1]
            box=box[np.logical_and(box_w>1,box_h)>1]
            box_data=np.zeros((len(box),5))
            box_data[:len(box)]=box
        return image_data,box_data
                
    def __getitem__(self,index):
        #一遍历这个类就会运行这个函数
        lines=self.train_lines
        num=self.train_batches
        index=index%num
        if self.is_train:
            img,y=self.get_random_data(lines[index],self.image_size[0:2])
        else:
            img,y=self.get_random_data(lines[index],self.image_size[0:2],False)

        #img:[416,416,3]未变换轨道的图片
        #y为(?,5)
        
        if len(y)!=0:
            #从坐标转为0到1的百分比
            boxes=np.array(y[:,:4],dtype=np.float32)
            boxes[:,0]=boxes[:,0]/self.image_size[1]
            boxes[:,1]=boxes[:,1]/self.image_size[0]
            boxes[:,2]=boxes[:,2]/self.image_size[1]
            boxes[:,3]=boxes[:,3]/self.image_size[0]
            #转为左上右下模式
            boxes=np.maximum(np.minimum(boxes,1),0)#防止有部分超出
            boxes[:,2]=boxes[:,2]-boxes[:,0]
            boxes[:,3]=boxes[:,1]-boxes[:,1]
            boxes[:,0]=boxes[:,0]+boxes[:,2]/2
            boxes[:,1]=boxes[:,2]+boxes[:,3]/2
            y=np.concatenate([boxes,y[:,-1:]],axis=-1)

        img=np.array(img,dtype=np.float32)
        img_normtran=np.transpose(img/255.0,(2,0,1))
        targets=np.array(y,dtype=np.float32)
        return img_normtran,targets
    
            
def yolo_dataset_collate(batch):
    images=[]
    bboxes=[]
    for img,box in batch:
        images.append(img)
        bboxes.append(box)
    images=np.array(images)
    return images,bboxes
