#!/usr/bin/env python3



from train_finetune_only import run_experiment


def main():
    config = {
        "task_name": "lunnan_h13_segamp_pairdip_ep150_bs40",
        "train_sample_path": "../data/lunnan/train_input_3d_inline_h13_segamp/",
        "mx_valid": True,
        "facies_3D": True,
        "data_shape": (256, 256, 416),
        "num_train_samples": 416,
        "loss_type": "gl_lunnan_h13_segamp_pairdip_ep150_bs40",
        "trans_epoch": None,
        "seg_first": True,
        "a3": (1, 1, 0.3),
        "pair_consistency_weight": 0.1,
        "pair_beta": 0.02,
        "pair_dip_path": "../data/lunnan/pair_dip_lunnan.npy",
        "seg_cross_slice": True,
        "pair_gap": 1,
        "frame_anchor_weight": 0.0,
        "seg_order_weight": 0.0,
        "boundary_weight": 0.0,
        "phase_weight": 0.0,
        "pred_local": "inline",
        "epochs": 150,
        "batch_size": 40,
        "lr": 1e-4,
        "scheduler_type": "cosine",
        "save_data": True,
        "CIGLoss_type": "L2",
        "mtl": False,
        "str_ort": False,
        "frame_part": False,
        "file_name": "lunnan",
        "ciglabel_dir": "../data/zxdata/datasets_ciglabel/",
    }
    run_experiment(config)


if __name__ == "__main__":
    main()
