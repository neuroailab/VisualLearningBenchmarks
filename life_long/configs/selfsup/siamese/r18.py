import copy
_base_ = '../../base.py'
# model settings
model = dict(
    type='Siamese',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV3',
        in_channels=512,
        hid_channels=512,
        out_channels=512,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              predictor=dict(type='NonLinearNeckV2',
                             in_channels=512, hid_channels=256,
                             out_channels=512, with_avg_pool=False)))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/imagenet/meta/train.txt'
data_train_root = 'data/imagenet/train'
dataset_type = 'SiameseDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3), # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0,
                )
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

data_val_nn_list = 'data/imagenet/meta/part_train_val_labeled.txt'
data_root = 'data/imagenet'
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=64,  # total 32*8=256
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val_nn=dict(
        type='NPIDNNDataset',
        data_source=dict(
            list_file=data_val_nn_list, root=data_root, **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val_nn'],
        initial=True,
        interval=1,
        imgs_per_gpu=64,
        workers_per_gpu=4,
        eval_param={}),
]
# optimizer
optimizer = dict(type='SGD', lr=0.05, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnealing', min_lr=0.)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
