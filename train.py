from ultralytics.models import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    model = YOLO(model='ultralytics/cfg/models/11/yolo11.yaml')
    model.load('yolo11n.pt')
    model.train(data='./dataset.yaml',      # 指定训练数据集的文件路径
                epochs=200,     # 设置训练轮数
                batch=32,       # 设置每个训练批次的大小
                device='0',     # 指定设备,'0'表示使用第一块GPU进行训练
                imgsz=640,      # 指定训练时使用的图像尺寸
                workers=8,      # 设置用于数据加载的线程数
                cache=False,    # 是否缓存数据集以加快后续训练速度
                amp=True,
                mosaic=False,
                project='runs/train',
                name='model_5')