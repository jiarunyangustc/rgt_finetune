#!/usr/bin/env python3



from train_finetune_only import run_experiment


def main():
    config = {
        "task_name": "zxdata_hr4_3d_xline256_lossonly_loraonly_ep200_bs40",
        "train_sample_path": "../data/zxdata/train_input_3d_mixdir256_gh_hr4_256_segqc/",
        "mx_valid": True,
        "facies_3D": True,
        "data_shape": (256, 256, 256),
        "loss_type": "gl_hr4_xline256_lossonly_loraonly_ep200_bs40",
        "trans_epoch": None,
        "seg_first": True,
        "a3": (1, 1, 0.3),
        "encoder_lora_only": True,
        "num_train_samples": 256,
        "pair_consistency_weight": 0.1,
        "pair_beta": 0.02,
        "pair_gap": 1,
        "frame_anchor_weight": 0.0,
        "seg_order_weight": 0.0,
        "pred_local": "xline",
        "epochs": 200,
        "batch_size": 40,
        "lr": 1e-4,
        "scheduler_type": "cosine",
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
