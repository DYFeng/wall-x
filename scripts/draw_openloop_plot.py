import os
import yaml
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs
from wall_x.model.model_utils import register_normalizers
import copy


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_horizon", type=int, default=10)
    parser.add_argument("--origin_action_dim", type=int, default=7)
    args = parser.parse_args()

    origin_action_dim = args.origin_action_dim

    # get train config
    # model_path = "/root/gpufree-data/output/train/libero_all/ckpt/0"
    # save_dir = "/root/gpufree-data/output/train/libero_all/open_loop"
    model_path = "/root/gpufree-data/output/train/so101-table-cleanup/ckpt/0"
    save_dir = "/root/gpufree-data/output/train/so101-table-cleanup/open_loop"

    action_tokenizer_path = "/root/gpufree-data/models/physical-intelligence/fast"
    save_dir = "/root/gpufree-data/output/train/libero_all/open_loop"
    path = f"{model_path}/config.yml"
    config = load_config(path)
    
    # 使用配置文件中的 action_horizon 或命令行参数
    pred_horizon = args.pred_horizon or config["data"].get("action_horizon", 10)

    normalizer_action, normalizer_propri = register_normalizers(config, model_path)

    # load model with customized robot config
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        model_path, train_config=config, action_tokenizer_path=action_tokenizer_path
    )

    model.set_normalizer(
        copy.deepcopy(normalizer_action), copy.deepcopy(normalizer_propri)
    )
    model.eval()
    model = model.to("cuda")
    model.to_bfloat16_for_selected_params()

    # get test dataloader
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})
    # 返回 1 个测试集（get_dataloader 默认返回测试集）
    dataset = load_test_dataset(
        config, lerobot_config, normalizer_action, normalizer_propri, seed=42
    )
    dataloader = dataset.get_dataloader()
    # dataloader = dataset.get_train_dataloader()

    total_frames = len(dataloader)

    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    # 使用模型实际的 action_dim
    action_dim = model.action_preprocessor.action_dim
    gt_traj = torch.zeros((total_frames, origin_action_dim))
    pred_traj = torch.zeros((total_frames, origin_action_dim))
    # 存储损失值
    loss_values = []

    # use tqdm to show the progress
    for idx, batch in tqdm(
        enumerate(dataloader), total=total_frames, desc="predicting"
    ):
        if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
            batch = batch.to("cuda")
            with torch.no_grad():
                outputs = model(
                    **batch,
                    action_dim=action_dim,
                    action_horizon=pred_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                )
                pred_action = outputs["predict_action"][:, :, :origin_action_dim]
                pred_traj[idx : idx + pred_horizon] = (
                    pred_action
                    .detach()
                    .cpu()
                    .squeeze(0)
                )

            # Denormalize ground truth actions
            gt_action_chunk = batch["action_chunk"][:, :, :origin_action_dim]
            dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
            denormalized_gt = (
                model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [lerobot_config.get("repo_id", "physical-intelligence/libero")],
                    dof_mask,
                ).squeeze(0)
            )
            gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()
            
            # 计算损失 (MSE)
            if pred_action is not None:
                mse_loss = torch.nn.functional.mse_loss(
                    pred_action.squeeze(0), 
                    denormalized_gt.unsqueeze(0).to(pred_action.device)
                )
                loss_values.append(mse_loss.item())
                print(f"Frame {idx}: MSE Loss = {mse_loss.item():.6f}")

    gt_traj_np = gt_traj.numpy()
    pred_traj_np = pred_traj.numpy()

    timesteps = gt_traj.shape[0]

    fig, axs = plt.subplots(
        origin_action_dim, 1, figsize=(15, 5 * origin_action_dim), sharex=True
    )
    fig.suptitle("Action Comparison for lerobot", fontsize=16)

    for i in range(origin_action_dim):
        axs[i].plot(range(timesteps), gt_traj_np[:, i], label="Ground Truth")
        axs[i].plot(range(timesteps), pred_traj_np[:, i], label="Prediction")
        axs[i].set_ylabel(f"Action Dim {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lerobot_comparison.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    # 计算平均损失
    if loss_values:
        avg_loss = sum(loss_values) / len(loss_values)
        print(f"\nAverage MSE Loss: {avg_loss:.6f}")
        
        # 保存损失值到文件
        loss_file = os.path.join(save_dir, "loss_values.txt")
        with open(loss_file, "w") as f:
            f.write(f"Average MSE Loss: {avg_loss:.6f}\n")
            f.write("\nFrame-wise Loss:\n")
            for i, loss in enumerate(loss_values):
                f.write(f"Frame {i * pred_horizon}: {loss:.6f}\n")
        print(f"Loss values saved to {loss_file}")
    
    plt.close()
