import os
USER_NAME = os.getlogin()

import openselfsup
FRMWK_REPO_PATH = os.path.dirname(openselfsup.__path__[0])
FS_BASE = f'/data1/{USER_NAME}/pub_clean_related/real_time_related'
FS_BASE = os.environ.get('FS_BASE', FS_BASE)

BASE_FOLDER = FS_BASE
STIM_PATH = './stim_paths_clean.pkl'
RESULT_FOLDER = os.path.join(FS_BASE, 'results')
IMAGENET_FOLDER = f'/data5/{USER_NAME}/Dataset/imagenet_raw/train/'
VGGFACE2_FOLDER = f'/data5/{USER_NAME}/Dataset/vgg_face2/jpgs/train/'

# SimCLR
SIMCLR_R18_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/simclr')
SIMCLR_R18_FACE_RDPD = os.path.join(SIMCLR_R18_FACE_RDPD_BASE, 'epoch_90.pth')    # perf: 0.797, dprime: 2.16

# SimCLR-More-MLPs
SIMCLR_MLP4_FACE_RDPD_EARLY_BASE = os.path.join(FS_BASE, 'models/simclr_more_mlps')
SIMCLR_MLP4_FACE_RDPD_EARLY = os.path.join(SIMCLR_MLP4_FACE_RDPD_EARLY_BASE, 'epoch_80.pth') # perf: 0.827, dprime: 1.92

# SimCLR-ResNet50
SIMCLR_R50_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/simclr_resnet50')
SIMCLR_R50_FACE_RDPD = os.path.join(SIMCLR_R50_FACE_RDPD_BASE, 'epoch_60.pth')    # 0.79, dp=1.75, lr=0.1

# DINO
DINO_MLP3_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/dino')
DINO_MLP3_FACE_RDPD = os.path.join(DINO_MLP3_FACE_RDPD_BASE, 'epoch_100.pth') # 0.727, dprime: 1.78

# DINONeg
DINONEG_MLP3_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/dinoneg')
DINONEG_MLP3_FACE_RDPD = os.path.join(DINONEG_MLP3_FACE_RDPD_BASE, 'epoch_100.pth')    # 0.77 6.6e-4

# MAE
MAE_VIT_S_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/mae')
MAE_VIT_S_FACE_RDPD = os.path.join(MAE_VIT_S_FACE_RDPD_BASE, 'epoch_210.pth')    # 0.59 (d' 0.8), LR 3.3e-4

# BYOL
BYOL_R18_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/byol')
BYOL_R18_FACE_RDPD = os.path.join(BYOL_R18_FACE_RDPD_BASE, 'epoch_20.pth')    # lr=0.3

# BYOL-More-MLPs
BYOL_MLP4_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/byol_more_mlps')
BYOL_MLP4_FACE_RDPD = os.path.join(BYOL_MLP4_FACE_RDPD_BASE, 'epoch_20.pth')    # 0.85 (face)

# BYOLNeg
BYOLNEG_R18_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/byolneg')
BYOLNEG_R18_FACE_RDPD = os.path.join(BYOLNEG_R18_FACE_RDPD_BASE, 'epoch_10.pth') # perf: 0.707, dprime: 1.62

# SwAV
SWAV_R18_FACE_RDPD_EARLY_BASE = os.path.join(FS_BASE, 'models/swav')
SWAV_R18_FACE_RDPD_EARLY = os.path.join(SWAV_R18_FACE_RDPD_EARLY_BASE, 'epoch_40.pth')    # 0.76, dp=1.74

# MoCo v2
MOCO_V2_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/mocov2')
MOCO_V2_FACE_RDPD = os.path.join(MOCO_V2_FACE_RDPD_BASE, 'epoch_90.pth')    # 0.857 (face)

# Barlow-Twins
BARLOW_TWINS_R18_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/barlow_twins')
BARLOW_TWINS_R18_FACE_RDPD = os.path.join(BARLOW_TWINS_R18_FACE_RDPD_BASE, 'epoch_130.pth') # perf: 0.902, dprime: 2.74

# SimSiam
SIAMESE_R18_FACE_RDPD_BASE = os.path.join(FS_BASE, 'models/simsiam')
SIAMESE_R18_FACE_RDPD = os.path.join(SIAMESE_R18_FACE_RDPD_BASE, 'epoch_30.pth')    # 0.75 (face)
