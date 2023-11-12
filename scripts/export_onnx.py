import os
import sys

import numpy as np
import torch

# 将项目根目录添加到 python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# noqa: E402
from model.net import PolicyValueNet


def export():
    # Model trained on 9x9 board according to state dict mismatch
    width = 9
    height = 9
    model_path = os.path.join(project_root, "outputs", "model_fast_final.pth")

    device = torch.device("cpu")
    model = PolicyValueNet(width, height)

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        # map_location='cpu' is important
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()

    # 输入形状: (batch_size, 4, height, width)
    # 4 个通道: [player_stones, opponent_stones, last_move, is_first_player]
    dummy_input = torch.randn(1, 4, height, width, device=device)

    # 输出路径
    output_dir = os.path.join(project_root, "web")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "model.onnx")

    print("Exporting to ONNX...")
    # 强制将权重保存到单个 ONNX 文件中，避免生成 external data
    # 对于小模型（<2GB），这通常是更好的选择，特别是 web 环境
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["log_probs", "value"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "log_probs": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
        opset_version=17,
        export_params=True,  # Store the trained parameter weights inside the model file
        do_constant_folding=True,  # Whether to execute constant folding for optimization
    )

    # 后处理：如果生成了外部数据文件，强制合并到 ONNX 主文件中
    import onnx

    print("Checking for external data...")
    onnx_model = onnx.load(output_path, load_external_data=True)

    # 检查是否使用了 external data
    use_external_data = False
    for initializer in onnx_model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            use_external_data = True
            break

    if use_external_data or os.path.exists(output_path + ".data"):
        print("External data detected. Merging into single ONNX file...")

        onnx.save(onnx_model, output_path)

        # 清理 .data 文件
        data_file = output_path + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)
            print(f"Removed external data file: {data_file}")

    print(f"Successfully exported model to {output_path}")


if __name__ == "__main__":
    export()
