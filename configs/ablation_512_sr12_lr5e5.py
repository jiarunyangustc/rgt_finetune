#!/usr/bin/env python3
# 单变量对照：与 pairw01 唯一差别 = pair loss 换 dip 补偿版（δ 场来自 PWD）。
# 目的：验证"沿构造方向一致"一般式在缓倾工区退化等价于同位置版（论文防御用）。
# 预期：与 pairw01 打平（|δ| med 0.19px, p99 1.0px, warp≈恒等的小扰动）。
from train_finetune_only import run_experiment


def main():
    config = {
        "task_name": "zxdata_hr4_3d_xline_segqc3_pairdip_a03_ep150_bs40",
        "train_sample_path": "../data/zxdata/train_input_3d_xline_gh_hr4_512_mod_seg_strongaxis_segqc3/",
        "mx_valid": True,
        "facies_3D": True,
        "data_shape": (512, 512, 512),
        "loss_type": "gl_hr4_segqc3_pairdip_a03_ep150_bs40",
        "trans_epoch": None,
        "seg_first": True,
        "a3": (1, 1, 0.3),
        "pair_consistency_weight": 0.1,
        "pair_dip_path": "../data/zxdata/pair_dip_512.npy",
        "frame_anchor_weight": 0.0,
        "seg_order_weight": 0.0,
        "boundary_weight": 0.0,
        "phase_weight": 0.0,
        "pred_local": "xline",
        "epochs": 150,
        "batch_size": 40,
        "lr": 5e-5,
        "scheduler_type": "cosine",
        "qc_gh_path": "../data/zxdata/ux_gh_256x256x256.dat",
        "qc_slice_step": 32,
        "qc_levels": 40,
        "save_data": True,
        "CIGLoss_type": "L2",
        "mtl": False,
        "str_ort": False,
        "frame_part": False,
        "file_name": "zxdata",
        "ciglabel_dir": "../data/zxdata/datasets_ciglabel/",
    }
    run_experiment(config)


if __name__ == "__main__":
    main()
