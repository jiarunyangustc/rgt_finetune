import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# The original trainer exposes several utilities through these modules.
from utils import *
from draw import *
from models_glp.model import GLPDepth_add_gradient

# ==========================================================
# Section-wise inference and 3-D assembly
# ==========================================================
def pred_dict_2d23d(model, samples):
    """Predict RGT for a sequence of section dictionaries."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    pred_samples = []

    print(f">>> Running inference on {len(samples)} sections")
    with torch.no_grad():
        for i, sample_pred in enumerate(tqdm(samples)):
            data = mea_std_norm(sample_pred["seis"])
            data = torch.from_numpy(data).unsqueeze(0).float()

            frame = (sample_pred["frame"])
            frame = torch.from_numpy(frame).unsqueeze(0).float()

            data, frame = data.to(device), frame.to(device)

            # Match the three-channel input used during RGT pretraining.
            data = torch.cat((frame*10, data, data), dim=1)

            target_hr = model(data)

            target_hr = target_hr.cpu().squeeze(0).numpy()

            sample_pred["pred"] = target_hr / 10
            pred_samples.append(sample_pred)

    return pred_samples

# ==========================================================
# Save a complete experiment record
# ==========================================================
def save_task_config(config_dict, save_path):
    file_path = os.path.join(save_path, 'experiment_config.json')
    def default_converter(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False, default=default_converter)
    print(f">>> Experiment configuration saved to {file_path}")

# ==========================================================
# Main fine-tuning experiment
# ==========================================================
def run_experiment(config):
    # Paths and experiment metadata.
    task_name = config['task_name']
    checkpoint_root = config.get('checkpoint_root', 'checkpoints')
    checkpoint_path_ft = os.path.join(checkpoint_root, task_name)
    if not os.path.exists(checkpoint_path_ft):
        os.makedirs(checkpoint_path_ft)

    save_task_config(config, checkpoint_path_ft)

    # Reproducible random state and compute device.
    random_state = int(config.get('seed', 12314))
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    requested_device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('A CUDA device was requested but CUDA is not available')
    device = torch.device(requested_device)
    num_GPU = torch.cuda.device_count() if device.type == 'cuda' else 0
    if torch.cuda.is_available():
        print(f"Available GPUs: {num_GPU}")

    # Construct the per-section dataset.
    if config['pred_local'] == 'inline':
        seis_len = config['data_shape'][2]
    else:
        seis_len = config['data_shape'][1]
    # Explicitly set num_train_samples for datasets mixing multiple directions.
    n_train = config.get('num_train_samples') or seis_len
    samples_train = [f'{i}' for i in range(n_train)]
    current_seg_mode = '3d' if config['facies_3D'] else '2d'

    print(f">>> Building dataset for {config['task_name']}")
    train_data = build_dataset_cigfacies(
        samples_train,
        config['train_sample_path'],
        'Train',
        frame_part=False,
        mx_valid=config['mx_valid'],
        seg_mode=current_seg_mode
    )

    # Initialize the network and load the RGT-pretrained checkpoint.
    model = GLPDepth_add_gradient(max_depth=10,is_train=True,relu=False,use_lora=True, lora_rank=4)

    loss_type = "SSIM"
    session_name = '_'.join(["glp_frame_gr_norelu_newdata", loss_type,'seis2rgt'])

    default_pretrained = os.path.join(
        checkpoint_root,
        'glp_frame_gr_norelu_newdata_SSIM_seis2rgt',
        'checkpoint-epoch790.pth',
    )
    pretrained_path = config.get('pretrained_checkpoint', default_pretrained)
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            'RGT-pretrained checkpoint not found: '
            f'{pretrained_path}. Set pretrained_checkpoint in the configuration.'
        )
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {
        (name[7:] if name.startswith('module.') else name): value
        for name, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f">>> Loaded RGT-pretrained weights from {pretrained_path}")
    print(
        f"    Missing keys: {len(incompatible.missing_keys)}; "
        f"unexpected keys: {len(incompatible.unexpected_keys)}"
    )
    model = model.to(device)
    if num_GPU > 1 and config.get('data_parallel', True):
        model = torch.nn.DataParallel(model, device_ids=range(num_GPU))


    # Fine-tuning range: LoRA plus selected encoder convolutions by default.
    lora_only = config.get('encoder_lora_only', False)
    full_ft = config.get('full_finetune', False)
    sr3 = config.get('sr_open_stage3', False)
    for name, param in model.named_parameters():
        if full_ft:
            param.requires_grad = True
        elif 'lora' in name:
            param.requires_grad = True
        elif lora_only:
            param.requires_grad = False
        elif 'dwconv' in name and 'encoder' in name:
            param.requires_grad = True
        elif 'encoder' in name and '.sr.' in name:
            if ('block1.' in name or 'block2.' in name
                    or (sr3 and 'block3.' in name)):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
    if full_ft:
        print(">>> Full-network fine-tuning is enabled")
    if lora_only:
        print(">>> LoRA-only fine-tuning is enabled; encoder convolutions are frozen")

    trainable_params = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    trainable_total = sum(c for _, c in trainable_params)
    total = sum(p.numel() for p in model.parameters())
    print(f">>> Trainable parameters: {trainable_total:,} / {total:,} ({100*trainable_total/total:.2f}%)")
    lora_cnt = sum(c for n, c in trainable_params if 'lora' in n)
    sr_s1_cnt = sum(c for n, c in trainable_params if '.sr.' in n and 'block1.' in n)
    sr_s2_cnt = sum(c for n, c in trainable_params if '.sr.' in n and 'block2.' in n)
    sr_s3_cnt = sum(c for n, c in trainable_params if '.sr.' in n and 'block3.' in n)
    dw_cnt = sum(c for n, c in trainable_params if 'dwconv' in n)
    print(f"    LoRA: {lora_cnt:,}")
    print(f"    SR Conv (Stage1): {sr_s1_cnt:,}, SR Conv (Stage2): {sr_s2_cnt:,}, "
          f"SR Conv (Stage3): {sr_s3_cnt:,}")
    print(f"    DWConv (all stages): {dw_cnt:,}")

    # Assemble the training-loop parameters.
    param = {
        'loss': config['loss_type'],
        'checkpoint_path': checkpoint_path_ft,
        'mx_valid': config['mx_valid'],
        'trans_epoch': config.get('trans_epoch', 75),
        'seg_first': config.get('seg_first', True),
        'a3': config.get('a3', (1, 1, 0.01)),
        'data_shape': config['data_shape'],
        'pred_local': config.get('pred_local', 'inline'),
        'facies_3D': config['facies_3D'],
        'epochs': config.get('epochs', 100),
        'batch_size': config.get('batch_size', 40),
        'lr': config.get('lr', 1e-4),
        'optimizer_type': config.get('optimizer_type', 'Adamw'),
        'weight_decay': config.get('weight_decay', 1e-3),
        'gamma': config.get('gamma', 0.9),
        'step_size': config.get('step_size', 50),
        'momentum': config.get('momentum', 0.8),
        'lr_factor': config.get('lr_factor', 0.5),
        'lr_patience': config.get('lr_patience', 8),
        'disp_inter': config.get('disp_inter', 2),
        'save_inter': config.get('save_inter', 10),
        'ol_fr1': config.get('ol_fr1', False),
        'ol_lora': config.get('ol_lora', False),

        'boundary_weight': config.get('boundary_weight', 0.0),
        'boundary_margin': config.get('boundary_margin', 0.5),
        'max_depth': config.get('max_depth', 10.0),
        'frame_anchor_weight': config.get('frame_anchor_weight', 0.0),
        'consistency_weight': config.get('consistency_weight', 0.0),
        'consistency_path': config.get('consistency_path', None),
        'phase_weight': config.get('phase_weight', 0.0),
        'phase_amp_percentile': config.get('phase_amp_percentile', 70.0),
        'phase_penalty': config.get('phase_penalty', 'l1'),
        'phase_penalty_scale': config.get('phase_penalty_scale', 0.3),
        'phase_warmup_epochs': config.get('phase_warmup_epochs', 0),
        'seg_order_weight': config.get('seg_order_weight', 0.0),
        'seg_order_warmup_epochs': config.get('seg_order_warmup_epochs', 0),
        'seg_order_min_points': config.get('seg_order_min_points', 5),

        'seg_depth_gamma': config.get('seg_depth_gamma', 0.0),

        'seg_cross_slice': config.get('seg_cross_slice', False),

        'pair_consistency_weight': config.get('pair_consistency_weight', 0.0),
        'pair_beta': config.get('pair_beta', 0.02),
        'pair_gap': config.get('pair_gap', 1),
        'pair_depth_gamma': config.get('pair_depth_gamma', 0.0),

        'pair_dip_path': config.get('pair_dip_path', ''),
        'pair_dip_scale': config.get('pair_dip_scale', 0.0),
        'shuffle': config.get('shuffle', False),

        'fault_pair_weight': config.get('fault_pair_weight', 0.0),
        'fault_pair_beta': config.get('fault_pair_beta', 0.03),
        'seg_order_min_depth_gap': config.get('seg_order_min_depth_gap', 4.0),
        'seg_order_margin': config.get('seg_order_margin', 0.02),
        'seg_order_max_segments': config.get('seg_order_max_segments', 128),
        'scheduler_type': config.get('scheduler_type', 'cosine'),

        'lr_lora': config.get('lr_lora', None),
        'lr_lora_mult': config.get('lr_lora_mult', 2.0),

        'qc_gh_path': config.get('qc_gh_path', None),
        'qc_slice_step': config.get('qc_slice_step', 32),
        'qc_levels': config.get('qc_levels', 40),
        'num_workers': config.get('num_workers', 4),
    }

    print(">>> Starting target-survey fine-tuning")
    model = finetune(
        param, model, train_data, criterion=None, input_attrs=["data"], output_attrs=["label"],
        plot=False, plot_epoch=2, transfer=True, facies=True, alp=None,
        mtl=config.get('mtl', False),
        str_ort=config.get('str_ort', False),
        frame_part=config.get('frame_part', False),
        save_data=config.get('save_data', True),
        CIGLoss_type=config.get('CIGLoss_type', 'L2'),
        file_name=config.get('file_name', 'zxdata'),
        ciglabel_dir=config.get('ciglabel_dir', '../data/zxdata/datasets_ciglabel/')
    )

    # Assemble predicted sections into a 3-D RGT volume.
    n1, n2, n3 = config['data_shape']
    pred_samples = pred_dict_2d23d(model, train_data)

    rgt_pred_in = np.zeros((n1, n2, n3), dtype=np.single)
    pred_local = config.get('pred_local', 'inline')
    for i in range(len(pred_samples)):
        if pred_local == 'inline':
            if i < n3:
                rgt_pred_in[:, :, i] = pred_samples[i]['pred'][0]
        else:
            if i < n2:
                rgt_pred_in[:, i, :] = pred_samples[i]['pred'][0]
    rgt_pred_in = rgt_pred_in.transpose()

    save_dir = config.get('output_dir', os.path.join('outputs', task_name))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config['loss_type']}.dat")
    print(f">>> Saving predicted RGT volume to {save_path}")
    rgt_pred_in.tofile(save_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f">>> Experiment {task_name} completed\n")
    return save_path

# ==========================================================

# ==========================================================
def main():
    """Run one experiment from a JSON configuration file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune the pretrained section-wise RGT network."
    )
    parser.add_argument(
        "config",
        help="JSON experiment configuration; Python paper configs can be run directly",
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    run_experiment(config)


if __name__ == "__main__":
    main()
