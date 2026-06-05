from ultralytics import YOLO

# 1. 加载模型
model = YOLO('C:\ultralytics-main\runs\detect\best-visdrone-200\weights\best.pt')

# 2. 对一张图片进行预测，并开启可视化
# source: 换成你的一张测试图片路径
model.predict(source='datasets/VisDrone/images/val/00001.jpg', visualize=True, save=True)