import cv2
import numpy as np
import os
from ultralytics import YOLO

# ================= 🔧 配置区域 (请修改这里) =================

# 1. 原始图片路径
IMG_PATH = r"C:\date\Visdrone2019\images\test2019\0000087_00299_d_0000002.jpg"

# 2. 基线模型路径 (Baseline)
MODEL_A_PATH = r'C:\ultralytics-main\runs\detect\yolo11n-visdrone2019-200\weights\best.pt'

# 3. 改进模型路径 (Ours)
MODEL_B_PATH = r'C:\ultralytics-main\runs\detect\best-visdrone-200\weights\best.pt'

# 4. 输出文件夹 (默认存当前目录)
OUTPUT_DIR = r"C:\visdrone_result_images"

# 5. 绘图参数
CONF_THRESHOLD = 0.25  # 置信度
LINE_WIDTH = 2  # 框的粗细 (VisDrone 建议 1 或 2)


# =======================================================

def imread_safe(path):
    """支持中文路径读取"""
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        return img
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None


def imwrite_safe(path, img):
    """支持中文路径保存"""
    try:
        cv2.imencode('.jpg', img)[1].tofile(path)
        print(f"✅ 已保存: {path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def run_separate_generation():
    # 0. 准备工作
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(IMG_PATH):
        print(f"❌ 找不到图片: {IMG_PATH}")
        return

    # 1. 读取原图
    print(f"📖 读取图片: {os.path.basename(IMG_PATH)}")
    origin_img = imread_safe(IMG_PATH)
    if origin_img is None: return

    # -------------------------------------------------
    # 2. 生成基线结果 (Baseline)
    # -------------------------------------------------
    if os.path.exists(MODEL_A_PATH):
        print("🚀 正在推理基线模型 (Baseline)...")
        model_a = YOLO(MODEL_A_PATH)
        res_a = model_a.predict(origin_img, conf=CONF_THRESHOLD, iou=0.45)[0]

        # 纯净绘图：无标签文字，无置信度数字，只有框
        img_a = res_a.plot(labels=False, conf=False, line_width=LINE_WIDTH)

        save_path_a = os.path.join(OUTPUT_DIR, "baseline_result.jpg")
        imwrite_safe(save_path_a, img_a)
    else:
        print(f"⚠️ 跳过基线模型 (找不到文件): {MODEL_A_PATH}")

    # -------------------------------------------------
    # 3. 生成改进结果 (Ours)
    # -------------------------------------------------
    if os.path.exists(MODEL_B_PATH):
        print("🚀 正在推理改进模型 (Ours)...")
        model_b = YOLO(MODEL_B_PATH)
        res_b = model_b.predict(origin_img, conf=CONF_THRESHOLD, iou=0.45)[0]

        # 纯净绘图
        img_b = res_b.plot(labels=False, conf=False, line_width=LINE_WIDTH)

        save_path_b = os.path.join(OUTPUT_DIR, "ours_result.jpg")
        imwrite_safe(save_path_b, img_b)
    else:
        print(f"⚠️ 跳过改进模型 (找不到文件): {MODEL_B_PATH}")

    print("\n🎉 全部完成！去看看这两个文件吧。")


if __name__ == '__main__':
    run_separate_generation()