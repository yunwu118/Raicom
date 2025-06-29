import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO(r'E:\YOLOv11\runs\train\model_5\weights\best.pt')

# 打开视频流
cap = cv2.VideoCapture(r'E:/Raicom/Drill/111.mp4')  # 可以是视频文件路径或摄像头索引

while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()

    if not ret:
        break  # 如果读取失败，退出循环

    # 进行预测
    results = model.predict(frame)

    # 提取检测结果
    for idx, result in enumerate(results):
        print(result)
        boxes = result.boxes.xyxy  # 边界框坐标
        scores = result.boxes.conf  # 置信度分数
        classes = result.boxes.cls  # 类别索引

        # 如果有类别名称，可以通过类别索引获取
        class_names = [model.names[int(cls)] for cls in classes]
        # 打印检测结果
        for box, score, class_name in zip(boxes, scores, class_names):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print("-------------------------")
            print(f"Class: {class_name}, Score: {score:.2f}, Box: {box}, Center: ({center_x:.2f}, {center_y:.2f})")

        # 可视化检测结果
        annotated_img = result.plot()

        # 显示图像
        cv2.imshow('Detected Image', annotated_img)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()
