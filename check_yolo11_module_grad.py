from collections import defaultdict

import torch

from ultralytics import YOLO

# =========================
# 1. 基本配置
# =========================

WEIGHTS = "C:/ultralytics-main/runs/detect/nassnet+ce_head-visdrone-200/weights/best.pt"  # 你的模型权重，也可以换成 runs/detect/train/weights/best.pt
DATA = "VisDrone.yaml"  # 你的数据集配置文件
IMGSZ = 640
BATCH = 2

# 想查看哪些层
# 例如 [2, 4, 6] 表示只看第 2、4、6 层
# None 表示查看所有 YOLO11 顶层模块
TARGET_LAYER_IDS = [1, 3]

# 每隔多少个 batch 打印一次
PRINT_EVERY = 1


# =========================
# 2. 递归提取输入中的 tensor
# =========================


def iter_tensors(x):
    """YOLO 中有些模块的输入可能是 tensor， 有些可能是 list/tuple，例如 Concat 模块。 这个函数用于把其中的 tensor 都取出来。.
    """
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for item in x:
            yield from iter_tensors(item)
    elif isinstance(x, dict):
        for item in x.values():
            yield from iter_tensors(item)


# =========================
# 3. 注册 hook：查看传给前一层的梯度
# =========================


def on_train_start(trainer):
    """训练开始时执行。 此时 Ultralytics 已经构建好了真正用于训练的 PyTorch 模型。.
    """
    net = trainer.model

    net._grad_input_norms = defaultdict(list)
    net._grad_handles = []
    net._grad_batch_count = 0

    print("\n================ YOLO11 Layers ================")
    for i, m in enumerate(net.model):
        print(f"{i}: {m.__class__.__name__}")
    print("===============================================\n")

    for i, module in enumerate(net.model):
        if TARGET_LAYER_IDS is not None and i not in TARGET_LAYER_IDS:
            continue

        layer_name = f"{i:02d}_{module.__class__.__name__}"

        def make_forward_hook(name):
            def forward_hook(module, inputs, output):
                """Forward 时拿到模块输入 inputs， 然后在输入 tensor 上注册 grad hook。 backward 时，这个 hook 会得到该输入 tensor 的梯度。
                这个梯度就是当前模块传给前一层的梯度。.
                """
                input_tensors = list(iter_tensors(inputs))

                for input_id, x in enumerate(input_tensors):
                    if not torch.is_tensor(x):
                        continue

                    if not x.requires_grad:
                        continue

                    def save_input_grad(grad, name=name, input_id=input_id):
                        grad_norm = grad.detach().float().norm(p=2).item()
                        grad_mean = grad.detach().float().abs().mean().item()
                        grad_max = grad.detach().float().abs().max().item()

                        net._grad_input_norms[name].append(
                            {
                                "input_id": input_id,
                                "shape": tuple(grad.shape),
                                "norm": grad_norm,
                                "mean": grad_mean,
                                "max": grad_max,
                            }
                        )

                    x.register_hook(save_input_grad)

            return forward_hook

        handle = module.register_forward_hook(make_forward_hook(layer_name))
        net._grad_handles.append(handle)

    print(f"[Gradient Debug] Registered hooks on {len(net._grad_handles)} modules.\n")


# =========================
# 4. 每个 batch 后打印梯度
# =========================


def on_train_batch_end(trainer):
    net = trainer.model
    net._grad_batch_count += 1

    if net._grad_batch_count % PRINT_EVERY != 0:
        return

    print(f"\n========== Gradient passed to previous layer | Batch {net._grad_batch_count} ==========")

    if len(net._grad_input_norms) == 0:
        print("No gradient was recorded.")
        return

    for layer_name, records in net._grad_input_norms.items():
        if len(records) == 0:
            continue

        norms = [r["norm"] for r in records]
        means = [r["mean"] for r in records]
        maxs = [r["max"] for r in records]
        shape = records[-1]["shape"]

        print(
            f"{layer_name:20s} | "
            f"input_grad_norm = {sum(norms) / len(norms):.6e} | "
            f"mean_abs = {sum(means) / len(means):.6e} | "
            f"max_abs = {max(maxs):.6e} | "
            f"shape = {shape}"
        )

    print("=============================================================================\n")

    net._grad_input_norms.clear()


# =========================
# 5. 训练结束后移除 hook
# =========================


def on_train_end(trainer):
    net = trainer.model

    for h in getattr(net, "_grad_handles", []):
        h.remove()

    print("[Gradient Debug] Hooks removed.")


# =========================
# 6. 启动一次调试训练
# =========================

if __name__ == "__main__":
    model = YOLO(WEIGHTS)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_end", on_train_end)

    model.train(
        data=DATA,
        epochs=1,
        imgsz=IMGSZ,
        batch=BATCH,
        amp=False,  # 调试梯度时建议关闭 AMP，否则梯度可能被 GradScaler 放大
        workers=0,
    )
