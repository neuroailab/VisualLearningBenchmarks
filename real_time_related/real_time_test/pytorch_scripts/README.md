# Real-Time Benchmark

Before starting, download the provided materials and models from [link](http://life-long-real-time-neurips.s3.amazonaws.com/real_time_related.tar.gz) and extract the contents under a folder, which will be used as the `FS_BASE` (you may want to correspondingly modify `local_paths.py).

Next, prepare the ImageNet and the VGGFace2 datasets and modify the `IMAGENET_FOLDER` and `VGGFACE2_FOLDER` in the `local_paths.py` correspondingly. Please put these two folders under the same folder, which you can set the environment variable `IMG_FACE_ROOT` as.

Although VGGFace2 is currently taken offline by the authors, you may find this [link](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b) useful.

## SimCLR

Run the following codes:
```
for cond in build break switch ; do CUDA_VISIBLE_DEVICES=[4 gpu numbers] python -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM run_exp.py --result_folder [folder to save results] --init_lr [lr] --setting_func settings/real_video_it.py:simclr_face_rdpd_mix_${cond} --exp_bs_scale 0.5 --real_time_window_size 20.0 --resume ; done
```
This will run all three conditions for `W=20m, R=1:3`. For `W=0.5m`, change the value for `real_time_window_size` to `0.5`.
For `R=1:1`, change the value for `exp_bs_scale` to `1.0`.
For `R=3:1`, change the value for `exp_bs_scale` to `1.5`.


## SimCLR-More-MLPs
Replace the `simclr_face_rdpd` in the command with `simclr_mlp4_early_face_rdpd`.


## SimCLR-ResNet50
Replace the `simclr_face_rdpd` in the command with `simclr_r50_face_rdpd`.


## DINO
Replace the `simclr_face_rdpd` in the command with `dino_mlp3_face_rdpd`.


## DINONeg
Replace the `simclr_face_rdpd` in the command with `dinoneg_mlp3_face_rdpd`.


## MAE
Replace the `simclr_face_rdpd` in the command with `mae_face_rdpd_mae_same_crop`.


## BYOL
Replace the `simclr_face_rdpd` in the command with `byol_r18_face_rdpd`.


## BYOL-More-MLPs
Replace the `simclr_face_rdpd` in the command with `byol_mlp4_face_rdpd`.


## BYOLNeg
Replace the `simclr_face_rdpd` in the command with `byolneg_r18_face_rdpd`.


## SwAV
Replace the `simclr_face_rdpd` in the command with `swav_r18_face_rdpd_early`. Use only 2 gpus.


## MoCo v2
Replace the `simclr_face_rdpd` in the command with `moco_face_rdpd`.


## Barlow-Twins
Replace the `simclr_face_rdpd` in the command with `barlow_twins_r18_face`.


## SimSiam
Replace the `simclr_face_rdpd` in the command with `siamese_face_rdpd`.


## Check results
Use function `plot_three_cond_five_point_effects` in script `../unsup_plas/notebook_utils/utils.py`.
