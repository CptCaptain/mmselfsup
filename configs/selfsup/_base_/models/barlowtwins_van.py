# model settings
model = dict(
    type='BarlowTwins',
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True),
    backbone=dict(
        # type='VAN_Official',
        # embed_dims=[32, 64, 160, 256], # B0 dims
        # drop_rate=0.0,
        # drop_path_rate=0.1,
        # depths=[3, 3, 5, 2],
        # norm_cfg=norm_cfg,
        # init_cfg=pretrained,
        type='VAN',
    ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=8192,
        out_channels=8192,
        num_layers=3,
        with_last_bn=False,
        with_last_bn_affine=False,
        with_avg_pool=True,
        init_cfg=dict(
            type='Kaiming', distribution='uniform', layer=['Linear'])),
    head=dict(
        type='LatentCrossCorrelationHead',
        in_channels=8192,
        loss=dict(type='CrossCorrelationLoss')))
