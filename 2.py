import os

import pandas as pd

from ultralytics.utils.plotting import plot_results

# ================= 设置 =================
# 你的原始 CSV 文件路径
csv_path = r"C:\ultralytics-main\runs\detect\PE-YOLO-visdrone2019-200\results.csv"
# 你想保留的轮数
end_epoch = 200
# =======================================

# 1. 读取原始数据
# 有些 csv 可能会有空格，strip() 去除列名空格防止报错
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

# 2. 截取数据 (取前 end_epoch 行)
# 注意：如果你的 CSV 里包含表头，pandas会自动处理，这里直接切片即可
df_cut = df.iloc[:end_epoch]

# 3. 保存为新的临时 CSV 文件
# 这样不会覆盖你原始的数据，比较安全
temp_csv_path = "results_cut.csv"
df_cut.to_csv(temp_csv_path, index=False)

# 4. 调用 YOLO 官方绘图工具
# 它会根据 cut.csv 生成 results_cut.png
plot_results(temp_csv_path)

print("处理完成！")
print(f"截取后的数据已保存至: {os.path.abspath(temp_csv_path)}")
print(f"最终图片已生成至: {os.path.abspath('results_cut.png')}")
