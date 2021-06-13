
from PIL import Image

from yolo import yolo

if __name__=="__main__":
    yolo=yolo()#实例化yolo这个类

    while True:
        img=input('输入文件地址:')
        try:
            image=Image.open(img)
        except:
            print('图片不能打开，请重新输入地址!')
            continue
        else:
            r_image=yolo.detect_image(image)#用yolo里的detect_image方法
            r_image.show()
