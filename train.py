import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo3 import yolobody
from nets.yolo_training import weights_init,yololoss,losshistory
from utils.dataloader import yolodataset,yolo_dataset_collate
#进行训练，包括计算loss与权重文件的加载与保存
#初始化中加载先验框的函数
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors=f.readline()
    anchors=[float(x) for x in anchors.split(',')]
    #对读取到的先验框进行分组，两个数据为一小组，三小组为一个大组
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

#划分数据集
def divide(val_percentage,imglabel_path):
    with open (imglabel_path) as f:
        lines=f.readlines()
    np.random.seed(10101)#设计随机数种子
    np.random.shuffle(lines)#对每一行进行打乱
    np.random.seed(None)
    num_val=int(len(lines)*val_percentage)
    num_train=len(lines)-num_val
    return num_val,num_train,lines

#训练部分
def one_epoch_train(net,yolo_less,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,cuda):
    total_loss=0
    val_loss=0

    net.train()
    print('开始训练')

    for iteration,batch in enumerate(gen):
        #一个batch为一个列表，列表里装有两个array数组，一个为图片，另外一个为列表里装有多个目标
        if iteration>=epoch_size:
            break
        images,targets=batch[0],batch[1]
        with torch.no_grad():
            if cuda:
                images=torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets=[torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            else:
                images=torch.from_numpy(images).type(torch.FloatTensor)
                targets=[torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
        #images为图片，与网络输入格式一致
        #targets为目标，是一个二维数组，一维里有个五个量
        #梯度清零
        optimizer.zero_grad()
        #前向传播
        outputs=net(images)
        losses=[]
        num_pos_all=0
        #损失计算
        for i in range(3):
            loss_item,num_pos=yolo_loss(outputs[i],targets)
            losses.append(loss_item)
            num_pos_all+=num_pos
        loss=sum(losses)/num_pos_all
        #反向传播
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        #显示进度
        print('total_loss:',total_loss/(iteration+1))
    print('完成训练')
    net.eval()
    
    print('开始验证')
    for iteration,batch in enumerate(gen_val):
        #一个batch为一个列表，列表里装有两个array数组，一个为图片，另外一个为列表里装有多个目标
        if iteration>=epoch_size:
            break
        images,targets=batch[0],batch[1]
        with torch.no_grad():
            if cuda:
                images=torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets=[torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            else:
                images=torch.from_numpy(images).type(torch.FloatTensor)
                targets=[torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
        #images为图片，与网络输入格式一致
        #targets为目标，是一个二维数组，一维里有个五个量
        #梯度清零
        optimizer.zero_grad()
        #前向传播
        with torch.no_grad():
            outputs=net(images)
        losses=[]
        num_pos_all=0
        #损失计算
        for i in range(3):
            loss_item,num_pos=yolo_loss(outputs[i],targets)
            losses.append(loss_item)
            num_pos_all+=num_pos
        loss=sum(losses)/num_pos_all
        val_loss+=loss.item()
        #显示进度
        print('total_loss:',val_loss/(iteration+1))

    loss_history.append_loss(total_loss/(epoch_size+1),val_loss/(epoch_size_val+1))
    print('完成验证')
    print('epoch:'+str(epoch+1)+'/'+str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' %(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('保存数据中, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('保存完成')
        
if __name__=="__main__":
    #关于是否使用GPU
    Cuda=True
    #关于损失是否使用归一化
    loss_normalize=False
    #关于网络要求输入的照片大小
    input_shape=(416,416)
    #关于分类的数量
    num_classes=20
    #关于先验框的路径
    anchors_path='model_data/yolo_anchors.txt'
    anchors=get_anchors(anchors_path)
    #实例化一个模型并初始化权值
    model=yolobody(anchors,num_classes)
    weights_init(model)
    #读取之前保存的权值文件
    '''
    model_path='model_data/yolo_weights.pth'
    print('开始读取权值文件')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict=model.state_dict()
    pretrained_dict=torch.load(model_path,map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('权值文件已被读取')
    '''
    net=model.train()


    #如果使用显卡，则多线程工作，加快速度
    if Cuda:
        net=torch.nn.DataParallel(model)
        cudnn.benchmark=True
        net=net.cuda()
    #加载计算loss的类
    yolo_loss=yololoss(np.reshape(anchors,[-1,2]), num_classes, (input_shape[1], input_shape[0]), Cuda, loss_normalize)#yololoss是一个类([[10,29],[],[],[],[]],20,(416,416),cuda,normalize)
    loss_history=losshistory('logs/')
    #获得图片与标签的类
    imglabel_path='train.txt'

    #划分训练集与验证集
    val_percentage=0.1
    num_val,num_train,lines=divide(val_percentage,imglabel_path)


    #进行冻结式训练,将后一个true关掉为正常训练
    if False:
        #设置相关参数
        lr=1e-3#学习率
        batch_size=2#每个batch的大小
        init_epoch=0#起始世代
        freeze_epoch=10#冻结世代
        #设置学习率逐步下降的优化器
        optimizer=optim.Adam(net.parameters(),lr)#将网络相关参数与lr丢进优化器
        lr_scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)
        #训练集设置
        train_dataset=yolodataset(lines[:num_train],(input_shape[0],input_shape[1]),True)
        val_dataset=yolodataset(lines[num_train:],(input_shape[0],input_shape[1]),False)
        gen=DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=4,pin_memory=True,
                       drop_last=True,collate_fn=yolo_dataset_collate)
        gen_val=DataLoader(val_dataset,shuffle=True,batch_size=batch_size,num_workers=4,pin_memory=True,
                       drop_last=True,collate_fn=yolo_dataset_collate)
    

        for param in model.backbone.parameters():
            param.requires_grad=False
        #判断数据集是否足够
        epoch_size=num_train//batch_size
        epoch_size_val=num_val//batch_size
        if epoch_size==0 or epoch_size_val==0:
            raise ValueError('数据集过少！')
        #开始训练
        for epoch in range(init_epoch,freeze_epoch):
            one_epoch_train(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, freeze_epoch, Cuda)
            lr_scheduler.step()#优化器开始优化

    if True:
        #设置相关参数
        lr=1e-3#学习率
        batch_size=2#每个batch的大小
        freeze_epoch=0#冻结世代
        unfreeze_epoch=10#解冻世代
        #设置学习率逐步下降的优化器
        optimizer=optim.Adam(net.parameters(),lr)#将网络相关参数与lr丢进优化器
        lr_scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)
        #训练集设置
        train_dataset=yolodataset(lines[:num_train],(input_shape[0],input_shape[1]),True)
        val_dataset=yolodataset(lines[num_train:],(input_shape[0],input_shape[1]),False)
        gen=DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=4,pin_memory=True,
                       drop_last=True,collate_fn=yolo_dataset_collate)
        gen_val=DataLoader(val_dataset,shuffle=True,batch_size=batch_size,num_workers=4,pin_memory=True,
                       drop_last=True,collate_fn=yolo_dataset_collate)
        
        for param in model.backbone.parameters():
            param.requires_grad=True
        #判断数据集是否足够
        epoch_size=num_train//batch_size
        epoch_size_val=num_val//batch_size
        if epoch_size==0 or epoch_size_val==0:
            raise ValueError('数据集过少！')
        #开始训练
        for epoch in range(freeze_epoch,unfreeze_epoch):
            one_epoch_train(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, freeze_epoch, Cuda)
            lr_scheduler.step()#优化器开始优化
        

