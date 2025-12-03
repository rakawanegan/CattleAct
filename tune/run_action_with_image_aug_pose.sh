python3 tune/action_with_image_aug_pose.py --multirun \
    "augmentation.randaugment.num_ops=range(1, 6)" \
    "augmentation.randaugment.magnitude=range(5, 30)" \
    "augmentation.masking_from_skeleton.cutout_prob=choice(0.5, 0.7, 0.8, 0.9, 1.0)" \
    "augmentation.masking_from_skeleton.n_holes=range(3, 9)" \
    "augmentation.masking_from_skeleton.skip_label=choice(true, false)" \
    "augmentation.masking_from_skeleton.unuse_low_conf_skel=choice(true, false)"