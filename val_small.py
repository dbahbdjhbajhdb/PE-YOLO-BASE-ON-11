from ultralytics import YOLO

# 这一行是必须的！Windows下没有它会报错
if __name__ == '__main__':
    # ---------------------------------------------------
    # 注意：下面的代码必须要有缩进 (Tab 或 4个空格)
    # ---------------------------------------------------

    # 1. 加载模型 (注意路径前加 r)
    model = YOLO(r'C:\ultralytics-main\runs\detect\yolo11(n)-visdrone2019-300\weights\best.pt')

    # 2. 运行验证
    # workers=0 是个保底方案：如果还是报错，可以强制单进程运行
    # 但通常加上 if __name__... 就能解决
    print("开始验证...")
    results = model.val(data='VisDrone.yaml', split='val', save_json=True)

    print("验证完成！请查看上方日志中的 area= small 结果")