# Life-long Learning Benchmark

Largely forked from OpenSelfSup.

## Preparation

You need to first install this repo (`pip install -e ./`).

After that, prepare the ImageNet following the instructions in OpenSelfSup.

Then, download the SAYCam videos belonging to Sam and extract the frames using script `./dataset_miscs/infant_headcam/extract_cleaner_frames.py`


## Network Training

Codes to run training:

```
MODEL_SAVE_FOLDER=[your folder to save results] SAVE_REC_TO_FILE=1 CUDA_VISIBLE_DEVICES=[2 gpu device numbers] python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM framework_general.py --setting [specific_setting]
```

### SimCLR
Setting for `W=20m, R=1:3`: `./configs/new_pplns/simclr/sam_s112ep100.py:r18_cotr_m2_wd20m_eq3_aw5`.
Here 20m can be changed to 30s to get the `W=0.5m` setting. `eq3` is for `R=1:3`, `eq1` is for `R=1:1`, and `eqd3` is for `R=3:1`.

### SimCLR-More-MLPs
Change the `m2` in `r18_cotr_m2_wd20m_eq3_aw5` to `m4`.

### SimCLR-ResNet50
Setting for `W=20m, R=1:3`: `./configs/new_pplns/simclr/sam_s112ep100.py:r50_cotr_ct_wd20m_eq3_aw5`. ResNet50 takes 4 gpus.

### BYOL
Setting for `W=20m, R=1:3`: `./configs/new_pplns/byol/sam_s112ep100.py:r18_cotr_m2_wd20m_eq3_aw5`.

### BYOL-More-MLPs
Change the `m2` in `r18_cotr_m2_wd20m_eq3_aw5` to `m4`.

### BYOLNeg
Setting for `W=20m, R=1:3`: `./configs/new_pplns/byol/sam_s112ep100.py:r18_neg_cotr_ct_wd20m_eq3_aw5`.

### MoCo v2
Setting for `W=20m, R=1:3`: `./configs/new_pplns/moco/sam_s112ep100.py:r18_cotr_ct_wd20m_eq3_aw5`.

### SwAV
Setting for `W=20m, R=1:3`: `./configs/new_pplns/swav/sam_s112ep100.py:r18_cotr_ct_wd20m_eq3_aw5`.

### SimSiam
Setting for `W=20m, R=1:3`: `./configs/new_pplns/simsiam/saycam_sep.py:r18_sam_cotr_wd20m_eq3_aw5`.

### Barlow-Twins
Setting for `W=20m, R=1:3`: `./configs/new_pplns/barlow_twins/sam_s112ep100.py:r18_cotr_wd20m_eq3_aw5`.

### MAE
Setting for `W=20m, R=1:3`: `./configs/new_pplns/mae/saycam_sep.py:vits_cotr_wd20m_eq3_aw5`.

### DINO
Setting for `W=20m, R=1:3`: `./configs/new_pplns/dino/saycam_sep.py:vsn_sam_cotr_wd20m_eq3_aw5`.

### DINONeg
Setting for `W=20m, R=1:3`: `./configs/new_pplns/dino/saycam_sep.py:neg_vsn_sam_cotr_wd20m_eq3_aw5`.
