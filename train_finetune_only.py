import sys 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '2')
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# 请确保已导入你的自定义模块
from utils import * # 假设 mea_std_norm 在这里
from draw import *
from models_glp.model import GLPDepth_add_gradient

# ==========================================================
# 0. 推理函数 (复现 Code1 的 pred_dict_2d23d)
# ==========================================================
def pred_dict_2d23d(model, samples):
    """
    完全复现 code1 中的推理逻辑
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    pred_samples = []

    print(f">>> 正在进行推理 (Samples: {len(samples)})...")
    with torch.no_grad(): 
        for i, sample_pred in enumerate(tqdm(samples)):    
            # 1. 数据预处理
            data = mea_std_norm(sample_pred["seis"]) # 确保 utils 里有这个函数
            data = torch.from_numpy(data).unsqueeze(0).float()
            
            frame = (sample_pred["frame"])
            frame = torch.from_numpy(frame).unsqueeze(0).float()        
            
            data, frame = data.to(device), frame.to(device)
            
            # 2. 构造输入 (Code1 逻辑: frame*10, data, data)
            # data = torch.cat((data, data,data), dim=1) # 原注释
            data = torch.cat((frame*10, data, data), dim=1)

            # 3. 模型预测
            target_hr = model(data) 

            # 4. 后处理 (Code1 逻辑: 除以 10)
            target_hr = target_hr.cpu().squeeze(0).numpy()   
            
            sample_pred["pred"] = target_hr / 10
            # sample_pred["frame"] =  sample_pred['frame'] # 原样保留
            
            pred_samples.append(sample_pred)
            
    return pred_samples

# ==========================================================
# 1. 辅助函数：将任务配置保存到本地
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
    print(f">>> 实验参数已备份至: {file_path}")

# ==========================================================
# 2. 核心训练任务封装 (只执行 Phase-1 LoRA 微调)
# ==========================================================
def run_experiment(config):
    # --- A. 路径初始化与参数备份 ---
    task_name = config['task_name']
    checkpoint_path_ft = os.path.join('checkpoints', task_name)
    if not os.path.exists(checkpoint_path_ft):
        os.makedirs(checkpoint_path_ft)
    
    save_task_config(config, checkpoint_path_ft)

    # --- B. 环境设置 ---
    random_state = int(config.get('seed', 12314))
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    # os.environ['CUDA_VISIBLE_DEVICES']= '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        num_GPU = torch.cuda.device_count()
        print(f"GPU 数量: {num_GPU}")

    # --- C. 动态构建数据集 ---
    if config['pred_local'] == 'inline':
        seis_len = config['data_shape'][2]
    else:
        seis_len = config['data_shape'][1]
    # num_train_samples: 混合方向等超出单方向切片数的数据集用；缺省=切片数（原行为）
    n_train = config.get('num_train_samples') or seis_len
    samples_train = [f'{i}' for i in range(n_train)]
    current_seg_mode = '3d' if config['facies_3D'] else '2d'
    
    print(f">>> 正在构建数据集: {config['task_name']}")
    train_data = build_dataset_cigfacies(
        samples_train,
        config['train_sample_path'], 
        'Train',
        frame_part=False,
        mx_valid=config['mx_valid'],      
        seg_mode=current_seg_mode        
    )

    # --- D. 模型初始化 ---
    model = GLPDepth_add_gradient(max_depth=10,is_train=True,relu=False,use_lora=True, lora_rank=4)

    loss_type = "SSIM"
    session_name = '_'.join(["glp_frame_gr_norelu_newdata", loss_type,'seis2rgt'])

    # 并行模式
    if num_GPU >= 1:
        model = torch.nn.DataParallel(model, device_ids=range(num_GPU)).to(device)
    else:
        model = model.to(device)
        
    # 模型保存路径    
    checkpoint_path = os.path.join('checkpoints', session_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print(f"模型读取路径: {checkpoint_path}")

    checkpoint_path = os.path.join('checkpoints','glp_frame_gr_norelu_newdata_SSIM_seis2rgt')
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'checkpoint-epoch790.pth'))['state_dict'], strict = False)
    print(f"模型读取路径: {checkpoint_path}")


    # --- 冻结策略：LoRA + 分辨率敏感的 Conv（Stage1/2 的 SR + 所有 DWConv）---
    # 设计考量：
    #   Stage 1 (sr_ratio=8, 786K) + Stage 2 (sr_ratio=4, 2.1M) 是分辨率迁移的主要痛点
    #   Stage 3 的 SR Conv 虽有 11M 参数，但 sr_ratio=2 变化较缓，解冻易破坏预训练表征
    #   DWConv 总参数仅约 456K，且承担 MiT 的隐式位置编码，全部解冻
    lora_only = config.get('encoder_lora_only', False)   # True: 纯 LoRA, conv 全部冻结
    full_ft = config.get('full_finetune', False)         # True: 全量微调, 所有参数解冻
    sr3 = config.get('sr_open_stage3', False)            # True: SRConv 额外放开 stage3(消融)
    for name, param in model.named_parameters():
        if full_ft:
            param.requires_grad = True
        elif 'lora' in name:
            param.requires_grad = True
        elif lora_only:
            param.requires_grad = False
        elif 'dwconv' in name and 'encoder' in name:
            # 所有 stage 的 DWConv 都解冻（参数量小，对分辨率敏感）
            param.requires_grad = True
        elif 'encoder' in name and '.sr.' in name:
            # 缺省只解冻 Stage 1/2 的 SR Conv; sr_open_stage3 时加 Stage 3
            if ('block1.' in name or 'block2.' in name
                    or (sr3 and 'block3.' in name)):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
    if full_ft:
        print(">>> full_finetune=True: 全部参数解冻 (全量微调基线)")
    if lora_only:
        print(">>> encoder_lora_only=True: 仅 LoRA 可训练, SR/DWConv 冻结")

    # 打印可训练参数概况，确认改动生效
    trainable_params = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    trainable_total = sum(c for _, c in trainable_params)
    total = sum(p.numel() for p in model.parameters())
    print(f">>> 可训练参数: {trainable_total:,} / {total:,} ({100*trainable_total/total:.4f}%)")
    # 按模块分组统计
    lora_cnt = sum(c for n, c in trainable_params if 'lora' in n)
    sr_s1_cnt = sum(c for n, c in trainable_params if '.sr.' in n and 'block1.' in n)
    sr_s2_cnt = sum(c for n, c in trainable_params if '.sr.' in n and 'block2.' in n)
    sr_s3_cnt = sum(c for n, c in trainable_params if '.sr.' in n and 'block3.' in n)
    dw_cnt = sum(c for n, c in trainable_params if 'dwconv' in n)
    print(f"    LoRA: {lora_cnt:,}")
    print(f"    SR Conv (Stage1): {sr_s1_cnt:,}, SR Conv (Stage2): {sr_s2_cnt:,}, "
          f"SR Conv (Stage3): {sr_s3_cnt:,}")
    print(f"    DWConv (all stages): {dw_cnt:,}")

    # --- E. 组装训练 Param 字典 ---
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
        # 新增：边界锚定 loss 与 scheduler 控制
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
        # segment loss 深度加权（此前缺透传：2026-07-10 的 a03_segdw2 实为 a03 复刻）
        'seg_depth_gamma': config.get('seg_depth_gamma', 0.0),
        # 跨切片 segment 分组(batch 内同全局 ID 共用中心, 要求 shuffle=False)
        'seg_cross_slice': config.get('seg_cross_slice', False),
        # 跨切片成对一致性
        'pair_consistency_weight': config.get('pair_consistency_weight', 0.0),
        'pair_beta': config.get('pair_beta', 0.02),
        'pair_gap': config.get('pair_gap', 1),
        'pair_depth_gamma': config.get('pair_depth_gamma', 0.0),
        # dip 补偿版 pair（δ 场路径, 空=同位置版; scale 0=自动取 pair_gap）
        'pair_dip_path': config.get('pair_dip_path', ''),
        'pair_dip_scale': config.get('pair_dip_scale', 0.0),
        'shuffle': config.get('shuffle', False),
        # 跨断层配对锚点 loss（打印复用 frame_anchor 字段）
        'fault_pair_weight': config.get('fault_pair_weight', 0.0),
        'fault_pair_beta': config.get('fault_pair_beta', 0.03),
        'seg_order_min_depth_gap': config.get('seg_order_min_depth_gap', 4.0),
        'seg_order_margin': config.get('seg_order_margin', 0.02),
        'seg_order_max_segments': config.get('seg_order_max_segments', 128),
        'scheduler_type': config.get('scheduler_type', 'cosine'),
        # LoRA 组独立 LR（None = base_lr × lr_lora_mult，与旧 run 行为一致）
        'lr_lora': config.get('lr_lora', None),
        'lr_lora_mult': config.get('lr_lora_mult', 2.0),
        # RGT 层位质控（只评价不训练，None = 不启用）
        'qc_gh_path': config.get('qc_gh_path', None),
        'qc_slice_step': config.get('qc_slice_step', 32),
        'qc_levels': config.get('qc_levels', 40),
    }

    # --- F. 执行训练 (Phase-1: LoRA 微调) ---
    print(f">>> 开始 LoRA 微调...")
    model = finetune(
        param, model, train_data, criterion=None, input_attrs=["data"], output_attrs=["label"],
        plot=False, plot_epoch=2, transfer=True, facies=True, alp=None,
        mtl=config.get('mtl', False),
        str_ort=config.get('str_ort', False),
        frame_part=config.get('frame_part', False),
        save_data=config.get('save_data', True),
        CIGLoss_type=config.get('CIGLoss_type', 'L2'),
        file_name=config.get('file_name', 'zxdata'), # 保持 code1 逻辑
        ciglabel_dir=config.get('ciglabel_dir', '../data/zxdata/datasets_ciglabel/')
    )

    # --- G. 推理与保存 ---
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

    save_dir = os.path.join('..', 'data', 'zxdata')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config['loss_type']}.dat")
    print(f">>> 保存预测结果至: {save_path}")
    rgt_pred_in.tofile(save_path)

    # --- H. 释放资源 ---
    del model
    torch.cuda.empty_cache()
    print(f">>> 实验 {task_name} 全部完成。\n")

# ==========================================================
# 3. 实验配置列表
# ==========================================================
if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']= '1'
    n1,n2,n3 = 256,256,256
    all_experiments = [
        # ============================================================
        # 实验 B: LoRA + SR(Stage1/2) + DWConv 解冻
        # 配套改动：
        #   - lr 1e-5 → 5e-5（之前 1e-5 看起来明显欠拟合，曲线极平缓）
        #   - epochs 400 → 200（后 200 epoch 边际收益过低）
        #   - trans_epoch 75 → 20（前期 facies 阶段不需要那么久）
        #   - 启用 boundary_weight=0.1（软边界锚定，抑制累加误差和平凡解）
        #   - scheduler_type='cosine'（让 LR 真正衰减）
        # ============================================================
        # {
        #     "task_name": "zxdata_512_LoRA_SR12_DW_lr5e5_segf_epoch75_cosine_bnd01_hr9",
        #     "train_sample_path": '../data/zxdata/train_input_3d_xline_9hor_norm_hr4_512/',
        #     "mx_valid": True,
        #     "facies_3D": True,
        #     "data_shape": ( n1,n2,n3 ),
        #     "loss_type": "gl_hr4_inline_mini200",
        #     "trans_epoch": 75,        # 75 → 20
        #     "seg_first": True,
        #     "a3": (1, 1, 0.01),
        #     "pred_local": 'xline',
        #     "epochs": 200,            # 400 → 200
        #     "batch_size": 20,
        #     "lr": 5e-5,               # 1e-5 → 5e-5

        #     # 新增的边界锚定参数
        #     "boundary_weight": 0.1,   # 软边界锚定权重
        #     "boundary_margin": 0.5,   # 容忍 margin
        #     "max_depth": 10.0,        # RGT 顶到底的目标范围
        #     "scheduler_type": "cosine",  # 让 LR 真正衰减

        #     # finetune 参数（保持不变）
        #     "save_data": True,
        #     "CIGLoss_type": "L2",
        #     "mtl": False,
        #     "str_ort": False,
        #     "frame_part": False,
        #     "file_name": "zxdata",
        #     "ciglabel_dir": "../data/zxdata/datasets_ciglabel/"
        # },
        # {
        #     "task_name": "zxdata_512_LoRA_SR12_DW_lr5e5_cosine_bnd01_hr9_hrf75",
        #     "train_sample_path": '../data/zxdata/train_input_3d_xline_9hor_norm_hr4_512/',
        #     "mx_valid": True,
        #     "facies_3D": True,
        #     "data_shape": ( n1,n2,n3 ),
        #     "loss_type": "gl_hr4_inline_mini200",
        #     "trans_epoch": 75,        # 75 → 20
        #     "seg_first": False,
        #     "a3": (1, 1, 0.01),
        #     "pred_local": 'xline',
        #     "epochs": 200,            # 400 → 200
        #     "batch_size": 20,
        #     "lr": 5e-5,               # 1e-5 → 5e-5

        #     # 新增的边界锚定参数
        #     "boundary_weight": 0.1,   # 软边界锚定权重
        #     "boundary_margin": 0.5,   # 容忍 margin
        #     "max_depth": 10.0,        # RGT 顶到底的目标范围
        #     "scheduler_type": "cosine",  # 让 LR 真正衰减

        #     # finetune 参数（保持不变）
        #     "save_data": True,
        #     "CIGLoss_type": "L2",
        #     "mtl": False,
        #     "str_ort": False,
        #     "frame_part": False,
        #     "file_name": "zxdata",
        #     "ciglabel_dir": "../data/zxdata/datasets_ciglabel/"
        # },
        {
            "task_name": "zxdata_512_LoRA_SR12_DW_lr5e5_cosine_bnd01_selecthor_both_fac01",
            "train_sample_path": '../data/zxdata/train_input_3d_xline_selecthor_hr4_512/',
            "mx_valid": True,
            "facies_3D": True,
            "data_shape": ( n1,n2,n3 ),
            "loss_type": "gl_hr4_inline_mini200",
            "trans_epoch": None,        # 75 → 20
            "seg_first": False,
            "a3": (1, 1, 0.1),
            "pred_local": 'xline',
            "epochs": 200,            # 400 → 200
            "batch_size": 20,
            "lr": 5e-5,               # 1e-5 → 5e-5

            # 新增的边界锚定参数
            "boundary_weight": 0.1,   # 软边界锚定权重
            "boundary_margin": 0.5,   # 容忍 margin
            "max_depth": 10.0,        # RGT 顶到底的目标范围
            "scheduler_type": "cosine",  # 让 LR 真正衰减

            # finetune 参数（保持不变）
            "save_data": True,
            "CIGLoss_type": "L2",
            "mtl": False,
            "str_ort": False,
            "frame_part": False,
            "file_name": "zxdata",
            "ciglabel_dir": "../data/zxdata/datasets_ciglabel/"
        },
        {
            "task_name": "zxdata_512_LoRA_SR12_DW_lr5e5_cosine_bnd01_selecthor_both_fac001",
            "train_sample_path": '../data/zxdata/train_input_3d_xline_selecthor_hr4_512/',
            "mx_valid": True,
            "facies_3D": True,
            "data_shape": ( n1,n2,n3 ),
            "loss_type": "gl_hr4_inline_mini200",
            "trans_epoch": None,        # 75 → 20
            "seg_first": False,
            "a3": (1, 1, 0.01),
            "pred_local": 'xline',
            "epochs": 200,            # 400 → 200
            "batch_size": 20,
            "lr": 5e-5,               # 1e-5 → 5e-5

            # 新增的边界锚定参数
            "boundary_weight": 0.1,   # 软边界锚定权重
            "boundary_margin": 0.5,   # 容忍 margin
            "max_depth": 10.0,        # RGT 顶到底的目标范围
            "scheduler_type": "cosine",  # 让 LR 真正衰减

            # finetune 参数（保持不变）
            "save_data": True,
            "CIGLoss_type": "L2",
            "mtl": False,
            "str_ort": False,
            "frame_part": False,
            "file_name": "zxdata",
            "ciglabel_dir": "../data/zxdata/datasets_ciglabel/"
        },
        {
            "task_name": "zxdata_512_LoRA_SR12_DW_lr5e5_cosine_bnd01_selecthor_both_fac1",
            "train_sample_path": '../data/zxdata/train_input_3d_xline_selecthor_hr4_512/',
            "mx_valid": True,
            "facies_3D": True,
            "data_shape": ( n1,n2,n3 ),
            "loss_type": "gl_hr4_inline_mini200",
            "trans_epoch": None,        # 75 → 20
            "seg_first": False,
            "a3": (1, 1, 1),
            "pred_local": 'xline',
            "epochs": 200,            # 400 → 200
            "batch_size": 20,
            "lr": 5e-5,               # 1e-5 → 5e-5

            # 新增的边界锚定参数
            "boundary_weight": 0.1,   # 软边界锚定权重
            "boundary_margin": 0.5,   # 容忍 margin
            "max_depth": 10.0,        # RGT 顶到底的目标范围
            "scheduler_type": "cosine",  # 让 LR 真正衰减

            # finetune 参数（保持不变）
            "save_data": True,
            "CIGLoss_type": "L2",
            "mtl": False,
            "str_ort": False,
            "frame_part": False,
            "file_name": "zxdata",
            "ciglabel_dir": "../data/zxdata/datasets_ciglabel/"
        },
        # ============================================================
        # （可选）实验 A baseline: 纯 LoRA（同 lr/epochs/trans_epoch，便于公平对比）
        # 把 task_name 留出，需要时取消注释跑
        # ============================================================
        # {
        #     "task_name": "zxdata_512_LoRA_only_lr5e5_cosine_bnd01",
        #     "train_sample_path": '../data/zxdata/train_input_3d_xline_mini5000_max100000_norm_hr4_512/',
        #     "mx_valid": True,
        #     "facies_3D": True,
        #     "data_shape": ( n1,n2,n3 ),
        #     "loss_type": "gl_hr4_inline_mini200",
        #     "trans_epoch": 20,
        #     "seg_first": True,
        #     "a3": (1, 1, 0.01),
        #     "pred_local": 'xline',
        #     "epochs": 200,
        #     "batch_size": 20,
        #     "lr": 5e-5,
        #     "boundary_weight": 0.1,
        #     "boundary_margin": 0.5,
        #     "max_depth": 10.0,
        #     "scheduler_type": "cosine",
        #     "save_data": True,
        #     "CIGLoss_type": "L2",
        #     "mtl": False,
        #     "str_ort": False,
        #     "frame_part": False,
        #     "file_name": "zxdata",
        #     "ciglabel_dir": "../data/zxdata/datasets_ciglabel/",
        #     # 用一个标志告诉 run_experiment 这是 baseline，需要回退冻结策略
        #     "_baseline_lora_only": True,
        # },
        ]

    for task_cfg in all_experiments:
        try:
            run_experiment(task_cfg)
        except Exception as e:
            print(f"\n[ERROR] {task_cfg['task_name']} 失败: {e}\n")
            import traceback
            traceback.print_exc()
            continue