import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.model.qwen2_5_based.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers import AutoProcessor

# 加载模型配置
config_path = "/root/gpufree-data/models/x-square-robot/wall-oss-flow-v0.1/config.json"

model_config = Qwen2_5_VLConfig.from_pretrained(config_path,device="cpu")

# 加载处理器
processor = AutoProcessor.from_pretrained("/root/gpufree-data/models/x-square-robot/wall-oss-flow-v0.1",device="cpu")

# 初始化模型
model = Qwen2_5_VLMoEForAction(model_config, processor=processor)
print(model)
sys.exit(0)
# # 打印模型结构
# def print_module_structure(module, prefix="", max_depth=3):
#     if max_depth <= 0:
#         return
    
#     for name, child in module.named_children():
#         full_name = f"{prefix}.{name}" if prefix else name
#         print(full_name)
#         print_module_structure(child, full_name, max_depth-1)

# print("=== Model Structure ===")
# print_module_structure(model, max_depth=4)

# # 特别检查 self_attn 模块
# print("\n=== Checking self_attn modules ===")
# if hasattr(model, 'model') and hasattr(model.model, 'layers'):
#     for i, layer in enumerate(model.model.layers):
#         print(f"Layer {i}:")
#         if hasattr(layer, 'self_attn'):
#             print(f"  self_attn type: {type(layer.self_attn).__name__}")
#             # 打印 self_attn 的子模块
#             for name, child in layer.self_attn.named_children():
#                 print(f"    self_attn.{name}")