# training schedule for 20e
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = dict(
    type='OneCycleLR',
    eta_max=0.000714,
    total_steps=100,
    pct_start=0.03,
    anneal_strategy='linear',
    div_factor=10,
    final_div_factor=10)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.000714, betas=(0.937, 0.999), weight_decay=0.0005))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
