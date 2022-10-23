_base_ = '../../base.py'
# model settings
model = dict(
    type='NPID',
    pretrained=None,
    save_bank_labels=True,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='LinearNeck',
        in_channels=512,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.07),
    memory_bank=dict(
        type='SimpleMemory', length=1281167, feat_dim=128, momentum=0.5))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/imagenet/meta/train_labeled_64rep.txt'
data_train_root = 'data/imagenet/train'
data_val_list = 'data/imagenet/meta/val_labeled_new.txt'
data_val_root = 'data/imagenet/val'
data_val_nn_list = 'data/imagenet/meta/part_train_val_labeled.txt'
data_root = 'data/imagenet'
dataset_type = 'NPIDDataset'
dataset_val_type = 'NPIDNNDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=64,  # total 32*8
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_val_list, root=data_val_root, **data_source_cfg),
        pipeline=test_pipeline),
    val_nn=dict(
        type=dataset_val_type,
        data_source=dict(
            list_file=data_val_nn_list, root=data_root, **data_source_cfg),
        pipeline=test_pipeline))

# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=1,
        imgs_per_gpu=64,
        workers_per_gpu=4,
        eval_param={}),
    dict(
        type='ValidateHook',
        dataset=data['val_nn'],
        initial=True,
        interval=1,
        imgs_per_gpu=64,
        workers_per_gpu=4,
        mode_name='get_embdng',
        eval_param={}),
]
# optimizer
optimizer = dict(
    type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9, nesterov=False)
# learning policy
lr_config = dict(policy='step', step=[120, 160])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
