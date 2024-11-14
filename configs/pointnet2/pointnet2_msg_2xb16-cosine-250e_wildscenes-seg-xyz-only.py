_base_ = [
    '../_base_/datasets/wildscenes-3d.py', '../_base_/models/pointnet2_msg.py',
    '../_base_/schedules/seg-cosine-200e.py', '../_base_/default_runtime.py'
]
class_names = (
            "bush", # 0
            "dirt", # 1
            "fence", # 2
            "grass", # 3
            "gravel", # 4
            "log", # 5
            "mud", # 6
            "object", # 7
            "other-terrain", # 8
            "rock", # 9
            "structure", # 10
            "tree-foliage", # 11
            "tree-trunk", # 12
        )
# model settings
model = dict(
    backbone=dict(in_channels=3),  # only [xyz]
    decode_head=dict(
        num_classes=len(class_names),
        ignore_index=len(class_names),
        # `class_weight` is generated in data pre-processing, saved in
        # `data/scannet/seg_info/train_label_weight.npy`
        # you can copy paste the values here, or input the file path as
        # `class_weight=data/scannet/seg_info/train_label_weight.npy`
        loss_decode=dict(class_weight=None)),
    test_cfg=dict(
        num_points=12544,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24))

# dataset settings
# in this setting, we only use xyz as network input
# so we need to re-write all the data pipeline

num_points = 12544
backend_args = None
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2],  # only load xyz coordinates
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.5,
        ignore_index=len(class_names),
        use_normalized_coord=False,
        enlarge_size=0.2,
        min_unique_num=None),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        backend_args=backend_args),
    dict(
        # a wrapper in order to successfully call test function
        # actually we don't perform test-time-aug
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(batch_size=16, dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = test_dataloader

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))

# PointNet2-MSG needs longer training time than PointNet2-SSG
train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=5)
