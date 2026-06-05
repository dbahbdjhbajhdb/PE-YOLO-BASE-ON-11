import json
import os
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= 🔧 必须修改为测试集路径 =================

# 1. 测试集原始标签文件夹 (里面必须全是 .txt)
#    (注意：确认这里面有文件！如果是空的，说明你下载的是 Challenge 集，没法算分)
TEST_GT_DIR = r"C:\datasets\visdrone2019-coco\VisDrone2019-DET-test-dev\annotations"

# 2. 改进模型的【测试集】预测 JSON
#    (就是你刚刚跑出 mAP 18.6% 的那个文件夹里的 predictions.json)
TEST_PRED_JSON = r"C:\ultralytics-main\runs\detect\val7\predictions.json"


# ==========================================================

def fix_test_ids_and_eval(txt_dir, pred_json_path):
    print(f"🔍 正在读取预测文件: {pred_json_path}")

    if not os.path.exists(pred_json_path):
        print("❌ 找不到预测文件！")
        return

    with open(pred_json_path, 'r') as f:
        preds = json.load(f)

    if not preds:
        print("❌ 预测文件是空的！")
        return

    # 1. 获取预测文件中所有的 image_id
    #    这就是“标准答案”的 ID，我们必须跟它保持一模一样
    pred_image_ids = list(set(p['image_id'] for p in preds))
    pred_image_ids.sort()

    print(f"🕵️ YOLO 预测了 {len(pred_image_ids)} 张图片。")
    print(f"   ID 样例: {pred_image_ids[0]} (类型: {type(pred_image_ids[0])})")

    # 2. 开始构建真值 JSON
    dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # VisDrone 类别映射 (保持原始 ID 1-10)
    categories = {
        1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
        6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor"
    }
    for cid, name in categories.items():
        dataset['categories'].append({"id": cid, "name": name, "supercategory": "object"})

    ann_id = 0
    match_count = 0
    missing_txt_count = 0

    print("🔄 正在匹配真值标签...")

    for image_id in pred_image_ids:
        # 添加图片信息
        dataset['images'].append({
            "id": image_id,
            "file_name": str(image_id),  # 这里的 file_name 不重要，只要 ID 对就行
            "width": 1360, "height": 765
        })

        # 3. 寻找对应的 txt 文件
        #    无论 ID 是 "00001" 还是 "00001.jpg"，我们都只取文件名部分
        stem = os.path.splitext(str(image_id))[0]
        txt_path = os.path.join(txt_dir, stem + ".txt")

        if os.path.exists(txt_path):
            match_count += 1
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6: continue
                    try:
                        # VisDrone 原始格式: x,y,w,h,score,cls...
                        cls_raw = int(parts[5])

                        # 过滤掉不评测的类别 (0 和 11)
                        if cls_raw < 1 or cls_raw > 10: continue

                        dataset['annotations'].append({
                            "id": ann_id,
                            "image_id": image_id,  # 【关键】这里强制使用和预测一模一样的 ID
                            "category_id": cls_raw,  # 保持 1-10，不要减 1
                            "bbox": [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])],
                            "area": float(parts[2]) * float(parts[3]),
                            "iscrowd": 0
                        })
                        ann_id += 1
                    except ValueError:
                        continue
        else:
            missing_txt_count += 1

    print(f"✅ 真值生成完毕！")
    print(f"   - 成功匹配到 txt 的图片数: {match_count}")
    print(f"   - 没找到 txt 的图片数: {missing_txt_count}")

    if match_count == 0:
        print("❌ 严重错误：一张 txt 都没匹配上！请检查 TEST_GT_DIR 路径是否正确，或者里面是否有 txt 文件。")
        return

    # 保存临时真值文件
    gt_save_path = 'temp_test_gt_aligned.json'
    with open(gt_save_path, 'w') as f:
        json.dump(dataset, f)

    # 4. 调用 COCOEval 算分
    print(f"\n🚀 开始计算测试集 AP_small (使用强制对齐ID)...")
    try:
        cocoGt = COCO(gt_save_path)
        cocoDt = cocoGt.loadRes(pred_json_path)

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        stats = cocoEval.stats
        print("\n" + "=" * 40)
        print(f"🔥 【测试集最终结果 (Test-Dev)】")
        print(f"🏆 mAP (全尺寸): {stats[0] * 100:.2f}% (应该接近 18.6%)")
        print(f"🦐 AP_small:     {stats[3] * 100:.2f}% <--- 终于出来了！")
        print("=" * 40)

    except Exception as e:
        print(f"❌ 评测出错: {e}")


if __name__ == '__main__':
    fix_test_ids_and_eval(TEST_GT_DIR, TEST_PRED_JSON)