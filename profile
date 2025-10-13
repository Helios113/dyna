[2025-08-06 10:02:48,068][streaming.base.dataset][INFO] - Because `predownload` was not specified, it will default to 8*batch_size if batch_size is not None, otherwise 64.
[2025-08-06 10:02:48,256][composer.utils.reproducibility][INFO] - Setting seed to 3039535338
[2025-08-06 10:02:48,297][composer.trainer.trainer][INFO] - Run name: 1754470968-umber-owl
/nfs-share/pa511/code_bases/dyna_project/dyna/.venv/lib/python3.12/site-packages/composer/callbacks/speed_monitor.py:290: UserWarning: gpu_flop count not found for nvidia h100 nvl with precision=amp_bf16 so MFU cannot be calculated and reported. gpu_flops_available can be manually overridden by setting gpu_flops_available in SpeedMonitor or nvidia h100 nvl can be added to GPU_AVAILABLE_FLOPS in composer/callbacks/speed_monitor.py
  self.gpu_flops_available = get_gpu_flops_available(state)
/nfs-share/pa511/code_bases/dyna_project/dyna/.venv/lib/python3.12/site-packages/composer/trainer/trainer.py:1556: UserWarning: Specifying `eval_interval=100ba` without an `eval_dataloader` has no effect. If trying to run an evaluator, make sure `eval_dataloader` is specified. Otherwise, set `eval_interval` to 0 or default value 1.
  warnings.warn(
[2025-08-06 10:02:48,540][composer.trainer.trainer][INFO] - Stepping schedulers every batch. To step schedulers every epoch, set `step_schedulers_every_batch=False`.
[2025-08-06 10:02:48,541][composer.trainer.trainer][INFO] - Setting seed to 3039535338
[2025-08-06 10:02:48,541][composer.utils.reproducibility][INFO] - Setting seed to 3039535338
[2025-08-06 10:02:48,562][composer.trainer.trainer][INFO] - Using precision Precision.AMP_BF16
******************************
Config:
composer_commit_hash: None
composer_version: 0.31.0
node_name: unknown because NODENAME environment variable not set
num_gpus_per_node: 1
num_nodes: 1
rank_zero_seed: 3039535338
time/remaining_estimate_unit: hours

******************************
[2025-08-06 10:02:48,626][streaming.base.dataset][INFO] - Because `num_canonical_nodes` was not specified, and `shuffle_algo` is py1e, it will default to be equal to physical nodes.
[2025-08-06 10:02:48,626][streaming.base.dataset][INFO] - Because `shuffle_block_size` was not specified, it will default to max(4_000_000 // num_canonical_nodes, 1 << 18) if num_canonical_nodes is not None, otherwise 262144.
[W806 10:02:48.299357116 CPUAllocator.cpp:245] Memory block of unknown size was allocated before the profiling started, profiler results will not include the deallocation event
[batch=1/20]:
	 Train time/epoch: 0
	 Train time/batch: 0
	 Train time/sample: 0
	 Train time/batch_in_epoch: 0
	 Train time/sample_in_epoch: 0
	 Train time/token: 0
	 Train time/token_in_epoch: 0
	 Train memory/current_allocated_mem: 0.9814
	 Train memory/current_active_mem: 0.9814
	 Train memory/current_inactive_mem: 0.8117
	 Train memory/current_reserved_mem: 3.2317
	 Train memory/peak_allocated_mem: 3.1312
	 Train memory/peak_active_mem: 3.1312
	 Train memory/peak_inactive_mem: 1.0129
	 Train memory/peak_reserved_mem: 3.2317
	 Train memory/alloc_retries: 0
	 Train trainer/device_train_microbatch_size: 2
	 Train loss/train/total: 0.0053
	 Train metrics/train/LanguageCrossEntropy: 10.9476
	 Train metrics/train/LanguagePerplexity: 56817.2148
	 Train metrics/train/TokenAccuracy: 0.0005
	 Train time/train: 0.0050
	 Train time/val: 0.0000
	 Train time/total: 0.0050
	 Train lr-DecoupledAdamW/group0: 0.0000
	 Train expert_selection/ffn_layer: <wandb.sdk.data_types.image.Image object at 0x711e4ce7f800>
	 Train expert_selection/attn_o_layer: <wandb.sdk.data_types.image.Image object at 0x711e4d5d42f0>
	 Train expert_selection/attn_v_layer: <wandb.sdk.data_types.image.Image object at 0x711e4d62f590>
	 Train metrics/load_balance_attn_o: 0.0000
	 Train metrics/load_balance_attn_v: 0.0000
	 Train metrics/load_balance_ffn: 0.0000
	 Train metrics/load_balance_attn_total_o: 0.0000
	 Train metrics/load_balance_attn_total_v: 0.0000
	 Train metrics/load_balance_total_ffn: 0.0000
	 Train metrics/shanon_entropy: 10.6357
	 Train metrics/last_token_entropy: 10.6357
	 Train entropy/shanon_entropy: <wandb.sdk.data_types.image.Image object at 0x711e4cecd5e0>
	 Train entropy/last_token_entropy: <wandb.sdk.data_types.image.Image object at 0x711e4cfbdb20>
[batch=10/20]:
	 Train time/batch: 9
	 Train time/sample: 18
	 Train time/batch_in_epoch: 9
	 Train time/sample_in_epoch: 18
	 Train time/token: 18432
	 Train time/token_in_epoch: 18432
	 Train memory/current_allocated_mem: 1.6253
	 Train memory/current_active_mem: 1.6253
	 Train memory/current_inactive_mem: 1.0674
	 Train memory/current_reserved_mem: 4.0538
	 Train memory/peak_allocated_mem: 3.8172
	 Train memory/peak_active_mem: 3.8172
	 Train memory/peak_inactive_mem: 1.3315
	 Train memory/peak_reserved_mem: 4.0538
	 Train memory/alloc_retries: 0
	 Train trainer/device_train_microbatch_size: 2
	 Train loss/train/total: 0.0053
	 Train metrics/train/LanguageCrossEntropy: 10.8053
	 Train metrics/train/LanguagePerplexity: 49279.4531
	 Train metrics/train/TokenAccuracy: 0.0010
	 Train time/train: 0.0280
	 Train time/val: 0.0000
	 Train time/total: 0.0280
	 Train lr-DecoupledAdamW/group0: 0.0000
	 Train time/remaining_estimate: 0.0229
	 Train expert_selection/ffn_layer: <wandb.sdk.data_types.image.Image object at 0x711e4c9fa480>
	 Train expert_selection/attn_o_layer: <wandb.sdk.data_types.image.Image object at 0x711e4cd1c080>
	 Train expert_selection/attn_v_layer: <wandb.sdk.data_types.image.Image object at 0x711e4d81d580>
	 Train metrics/load_balance_attn_o: 0.0000
	 Train metrics/load_balance_attn_v: 0.0000
	 Train metrics/load_balance_ffn: 0.0000
	 Train metrics/load_balance_attn_total_o: 0.0000
	 Train metrics/load_balance_attn_total_v: 0.0000
	 Train metrics/load_balance_total_ffn: 0.0000
	 Train metrics/shanon_entropy: 10.6358
	 Train metrics/last_token_entropy: 10.6358
	 Train entropy/shanon_entropy: <wandb.sdk.data_types.image.Image object at 0x711e4c8bf0e0>
	 Train entropy/last_token_entropy: <wandb.sdk.data_types.image.Image object at 0x711e4d68c7d0>
	 Train l2_norm/moment/model.transformer.layers.0.attention.v: 0.0001
	 Train l2_norm/param/model.transformer.layers.0.attention.v: 51.1981
	 Train l2_norm/update/model.transformer.layers.0.attention.v: 0.0115
	 Train l2_norm/grad/model.transformer.layers.0.attention.v: 0.0002
	 Train l2_norm/moment/model.transformer.layers.0.attention.o: 0.0001
	 Train l2_norm/param/model.transformer.layers.0.attention.o: 57.4435
	 Train l2_norm/update/model.transformer.layers.0.attention.o: 0.0114
	 Train l2_norm/grad/model.transformer.layers.0.attention.o: 0.0002
	 Train l2_norm/moment/model.transformer.layers.0.attention.sel_v: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.attention.sel_v: 5.6424
	 Train l2_norm/update/model.transformer.layers.0.attention.sel_v: 0.0006
	 Train l2_norm/grad/model.transformer.layers.0.attention.sel_v: 0.0001
	 Train l2_norm/moment/model.transformer.layers.0.attention.sel_o: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.attention.sel_o: 5.6348
	 Train l2_norm/update/model.transformer.layers.0.attention.sel_o: 0.0006
	 Train l2_norm/grad/model.transformer.layers.0.attention.sel_o: 0.0001
	 Train l2_norm/moment/model.transformer.layers.0.attention.q.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.attention.q.weight: 18.1377
	 Train l2_norm/update/model.transformer.layers.0.attention.q.weight: 0.0034
	 Train l2_norm/grad/model.transformer.layers.0.attention.q.weight: 0.0001
	 Train l2_norm/moment/model.transformer.layers.0.attention.k.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.attention.k.weight: 18.0613
	 Train l2_norm/update/model.transformer.layers.0.attention.k.weight: 0.0034
	 Train l2_norm/grad/model.transformer.layers.0.attention.k.weight: 0.0001
	 Train l2_norm/moment/model.transformer.layers.0.ffn.keys: 0.0001
	 Train l2_norm/param/model.transformer.layers.0.ffn.keys: 140.9023
	 Train l2_norm/update/model.transformer.layers.0.ffn.keys: 0.0261
	 Train l2_norm/grad/model.transformer.layers.0.ffn.keys: 0.0002
	 Train l2_norm/moment/model.transformer.layers.0.ffn.values: 0.0005
	 Train l2_norm/param/model.transformer.layers.0.ffn.values: 20.3045
	 Train l2_norm/update/model.transformer.layers.0.ffn.values: 0.0338
	 Train l2_norm/grad/model.transformer.layers.0.ffn.values: 0.0016
	 Train l2_norm/moment/model.transformer.layers.0.ffn.expert_sel: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.ffn.expert_sel: 12.4396
	 Train l2_norm/update/model.transformer.layers.0.ffn.expert_sel: 0.0022
	 Train l2_norm/grad/model.transformer.layers.0.ffn.expert_sel: 0.0000
	 Train l2_norm/moment/model.transformer.layers.0.attn_pre.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.attn_pre.weight: 20.2977
	 Train l2_norm/update/model.transformer.layers.0.attn_pre.weight: 0.0002
	 Train l2_norm/grad/model.transformer.layers.0.attn_pre.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.0.ffn_pre.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.ffn_pre.weight: 20.2978
	 Train l2_norm/update/model.transformer.layers.0.ffn_pre.weight: 0.0002
	 Train l2_norm/grad/model.transformer.layers.0.ffn_pre.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.attention.v: 0.0001
	 Train l2_norm/param/model.transformer.layers.1.attention.v: 51.2626
	 Train l2_norm/update/model.transformer.layers.1.attention.v: 0.0119
	 Train l2_norm/grad/model.transformer.layers.1.attention.v: 0.0002
	 Train l2_norm/moment/model.transformer.layers.1.attention.o: 0.0001
	 Train l2_norm/param/model.transformer.layers.1.attention.o: 57.4358
	 Train l2_norm/update/model.transformer.layers.1.attention.o: 0.0117
	 Train l2_norm/grad/model.transformer.layers.1.attention.o: 0.0002
	 Train l2_norm/moment/model.transformer.layers.1.attention.sel_v: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attention.sel_v: 5.6853
	 Train l2_norm/update/model.transformer.layers.1.attention.sel_v: 0.0006
	 Train l2_norm/grad/model.transformer.layers.1.attention.sel_v: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.attention.sel_o: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attention.sel_o: 5.7290
	 Train l2_norm/update/model.transformer.layers.1.attention.sel_o: 0.0006
	 Train l2_norm/grad/model.transformer.layers.1.attention.sel_o: 0.0001
	 Train l2_norm/moment/model.transformer.layers.1.attention.q.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attention.q.weight: 18.1237
	 Train l2_norm/update/model.transformer.layers.1.attention.q.weight: 0.0034
	 Train l2_norm/grad/model.transformer.layers.1.attention.q.weight: 0.0001
	 Train l2_norm/moment/model.transformer.layers.1.attention.k.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attention.k.weight: 18.1608
	 Train l2_norm/update/model.transformer.layers.1.attention.k.weight: 0.0034
	 Train l2_norm/grad/model.transformer.layers.1.attention.k.weight: 0.0001
	 Train l2_norm/moment/model.transformer.layers.1.ffn.keys: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.ffn.keys: 140.8706
	 Train l2_norm/update/model.transformer.layers.1.ffn.keys: 0.0263
	 Train l2_norm/grad/model.transformer.layers.1.ffn.keys: 0.0002
	 Train l2_norm/moment/model.transformer.layers.1.ffn.values: 0.0004
	 Train l2_norm/param/model.transformer.layers.1.ffn.values: 20.3026
	 Train l2_norm/update/model.transformer.layers.1.ffn.values: 0.0346
	 Train l2_norm/grad/model.transformer.layers.1.ffn.values: 0.0013
	 Train l2_norm/moment/model.transformer.layers.1.ffn.expert_sel: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.ffn.expert_sel: 12.4756
	 Train l2_norm/update/model.transformer.layers.1.ffn.expert_sel: 0.0022
	 Train l2_norm/grad/model.transformer.layers.1.ffn.expert_sel: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.attn_pre.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attn_pre.weight: 20.2977
	 Train l2_norm/update/model.transformer.layers.1.attn_pre.weight: 0.0002
	 Train l2_norm/grad/model.transformer.layers.1.attn_pre.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.ffn_pre.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.ffn_pre.weight: 20.2978
	 Train l2_norm/update/model.transformer.layers.1.ffn_pre.weight: 0.0002
	 Train l2_norm/grad/model.transformer.layers.1.ffn_pre.weight: 0.0000
	 Train l2_norm/moment/model.embedding.weight: 0.0001
	 Train l2_norm/param/model.embedding.weight: 221.7130
	 Train l2_norm/update/model.embedding.weight: 0.0110
	 Train l2_norm/grad/model.embedding.weight: 0.0002
	 Train l2_norm/moment/model.lm_head.weight: 0.0002
	 Train l2_norm/param/model.lm_head.weight: 128.0163
	 Train l2_norm/update/model.lm_head.weight: 0.0271
	 Train l2_norm/grad/model.lm_head.weight: 0.0005
	 Train l2_norm/moment/model.lm_head.bias: 0.0000
	 Train l2_norm/param/model.lm_head.bias: 6.2700
	 Train l2_norm/update/model.lm_head.bias: 0.0032
	 Train l2_norm/grad/model.lm_head.bias: 0.0000
	 Train l2_norm/moment/model.out_norm.weight: 0.0000
	 Train l2_norm/param/model.out_norm.weight: 20.2964
	 Train l2_norm/update/model.out_norm.weight: 0.0004
	 Train l2_norm/grad/model.out_norm.weight: 0.0000
	 Train l2_norm/grad/global: 0.0022
[batch=20/20]:
	 Train time/batch: 19
	 Train time/sample: 38
	 Train time/batch_in_epoch: 19
	 Train time/sample_in_epoch: 38
	 Train time/token: 38912
	 Train time/token_in_epoch: 38912
	 Train memory/current_allocated_mem: 1.6341
	 Train memory/current_active_mem: 1.6341
	 Train memory/current_inactive_mem: 1.0691
	 Train memory/current_reserved_mem: 4.0643
	 Train memory/peak_allocated_mem: 3.8260
	 Train memory/peak_active_mem: 3.8260
	 Train memory/peak_inactive_mem: 1.4082
	 Train memory/peak_reserved_mem: 4.0643
	 Train memory/alloc_retries: 0
	 Train trainer/device_train_microbatch_size: 2
	 Train loss/train/total: 0.0051
	 Train metrics/train/LanguageCrossEntropy: 10.4162
	 Train metrics/train/LanguagePerplexity: 33397.8672
	 Train metrics/train/TokenAccuracy: 0.0515
	 Train throughput/batches_per_sec: 0.1085
	 Train throughput/samples_per_sec: 0.2170
	 Train throughput/device/batches_per_sec: 0.1085
	 Train throughput/device/samples_per_sec: 0.2170
	 Train throughput/tokens_per_sec: 222.2194
	 Train throughput/device/tokens_per_sec: 222.2194
	 Train time/train: 0.0536
	 Train time/val: 0.0000
	 Train time/total: 0.0536
	 Train lr-DecoupledAdamW/group0: 0.0001
	 Train time/remaining_estimate: 0.0000
	 Train expert_selection/ffn_layer: <wandb.sdk.data_types.image.Image object at 0x711e4be2d700>
	 Train expert_selection/attn_o_layer: <wandb.sdk.data_types.image.Image object at 0x711e4cc61970>
	 Train expert_selection/attn_v_layer: <wandb.sdk.data_types.image.Image object at 0x711e4cff32c0>
	 Train metrics/load_balance_attn_o: 0.0001
	 Train metrics/load_balance_attn_v: 0.0000
	 Train metrics/load_balance_ffn: 0.0000
	 Train metrics/load_balance_attn_total_o: 0.0000
	 Train metrics/load_balance_attn_total_v: 0.0000
	 Train metrics/load_balance_total_ffn: 0.0000
	 Train metrics/shanon_entropy: 10.6356
	 Train metrics/last_token_entropy: 10.6357
	 Train entropy/shanon_entropy: <wandb.sdk.data_types.image.Image object at 0x711e4d53fc20>
	 Train entropy/last_token_entropy: <wandb.sdk.data_types.image.Image object at 0x711e46d77b00>
	 Train l2_norm/moment/model.transformer.layers.0.attention.v: 0.0001
	 Train l2_norm/param/model.transformer.layers.0.attention.v: 51.1968
	 Train l2_norm/update/model.transformer.layers.0.attention.v: 0.0203
	 Train l2_norm/grad/model.transformer.layers.0.attention.v: 0.0002
	 Train l2_norm/moment/model.transformer.layers.0.attention.o: 0.0001
	 Train l2_norm/param/model.transformer.layers.0.attention.o: 57.4422
	 Train l2_norm/update/model.transformer.layers.0.attention.o: 0.0210
	 Train l2_norm/grad/model.transformer.layers.0.attention.o: 0.0002
	 Train l2_norm/moment/model.transformer.layers.0.attention.sel_v: 0.0001
	 Train l2_norm/param/model.transformer.layers.0.attention.sel_v: 5.6421
	 Train l2_norm/update/model.transformer.layers.0.attention.sel_v: 0.0018
	 Train l2_norm/grad/model.transformer.layers.0.attention.sel_v: 0.0001
	 Train l2_norm/moment/model.transformer.layers.0.attention.sel_o: 0.0001
	 Train l2_norm/param/model.transformer.layers.0.attention.sel_o: 5.6345
	 Train l2_norm/update/model.transformer.layers.0.attention.sel_o: 0.0018
	 Train l2_norm/grad/model.transformer.layers.0.attention.sel_o: 0.0001
	 Train l2_norm/moment/model.transformer.layers.0.attention.q.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.attention.q.weight: 18.1373
	 Train l2_norm/update/model.transformer.layers.0.attention.q.weight: 0.0050
	 Train l2_norm/grad/model.transformer.layers.0.attention.q.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.0.attention.k.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.attention.k.weight: 18.0609
	 Train l2_norm/update/model.transformer.layers.0.attention.k.weight: 0.0051
	 Train l2_norm/grad/model.transformer.layers.0.attention.k.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.0.ffn.keys: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.ffn.keys: 140.9023
	 Train l2_norm/update/model.transformer.layers.0.ffn.keys: 0.0398
	 Train l2_norm/grad/model.transformer.layers.0.ffn.keys: 0.0001
	 Train l2_norm/moment/model.transformer.layers.0.ffn.values: 0.0004
	 Train l2_norm/param/model.transformer.layers.0.ffn.values: 20.3091
	 Train l2_norm/update/model.transformer.layers.0.ffn.values: 0.0499
	 Train l2_norm/grad/model.transformer.layers.0.ffn.values: 0.0009
	 Train l2_norm/moment/model.transformer.layers.0.ffn.expert_sel: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.ffn.expert_sel: 12.4400
	 Train l2_norm/update/model.transformer.layers.0.ffn.expert_sel: 0.0037
	 Train l2_norm/grad/model.transformer.layers.0.ffn.expert_sel: 0.0000
	 Train l2_norm/moment/model.transformer.layers.0.attn_pre.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.attn_pre.weight: 20.2966
	 Train l2_norm/update/model.transformer.layers.0.attn_pre.weight: 0.0006
	 Train l2_norm/grad/model.transformer.layers.0.attn_pre.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.0.ffn_pre.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.0.ffn_pre.weight: 20.2979
	 Train l2_norm/update/model.transformer.layers.0.ffn_pre.weight: 0.0003
	 Train l2_norm/grad/model.transformer.layers.0.ffn_pre.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.attention.v: 0.0001
	 Train l2_norm/param/model.transformer.layers.1.attention.v: 51.2615
	 Train l2_norm/update/model.transformer.layers.1.attention.v: 0.0212
	 Train l2_norm/grad/model.transformer.layers.1.attention.v: 0.0002
	 Train l2_norm/moment/model.transformer.layers.1.attention.o: 0.0001
	 Train l2_norm/param/model.transformer.layers.1.attention.o: 57.4346
	 Train l2_norm/update/model.transformer.layers.1.attention.o: 0.0217
	 Train l2_norm/grad/model.transformer.layers.1.attention.o: 0.0002
	 Train l2_norm/moment/model.transformer.layers.1.attention.sel_v: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attention.sel_v: 5.6849
	 Train l2_norm/update/model.transformer.layers.1.attention.sel_v: 0.0016
	 Train l2_norm/grad/model.transformer.layers.1.attention.sel_v: 0.0001
	 Train l2_norm/moment/model.transformer.layers.1.attention.sel_o: 0.0001
	 Train l2_norm/param/model.transformer.layers.1.attention.sel_o: 5.7287
	 Train l2_norm/update/model.transformer.layers.1.attention.sel_o: 0.0018
	 Train l2_norm/grad/model.transformer.layers.1.attention.sel_o: 0.0001
	 Train l2_norm/moment/model.transformer.layers.1.attention.q.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attention.q.weight: 18.1232
	 Train l2_norm/update/model.transformer.layers.1.attention.q.weight: 0.0051
	 Train l2_norm/grad/model.transformer.layers.1.attention.q.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.attention.k.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attention.k.weight: 18.1603
	 Train l2_norm/update/model.transformer.layers.1.attention.k.weight: 0.0051
	 Train l2_norm/grad/model.transformer.layers.1.attention.k.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.ffn.keys: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.ffn.keys: 140.8709
	 Train l2_norm/update/model.transformer.layers.1.ffn.keys: 0.0411
	 Train l2_norm/grad/model.transformer.layers.1.ffn.keys: 0.0001
	 Train l2_norm/moment/model.transformer.layers.1.ffn.values: 0.0004
	 Train l2_norm/param/model.transformer.layers.1.ffn.values: 20.3076
	 Train l2_norm/update/model.transformer.layers.1.ffn.values: 0.0520
	 Train l2_norm/grad/model.transformer.layers.1.ffn.values: 0.0008
	 Train l2_norm/moment/model.transformer.layers.1.ffn.expert_sel: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.ffn.expert_sel: 12.4760
	 Train l2_norm/update/model.transformer.layers.1.ffn.expert_sel: 0.0035
	 Train l2_norm/grad/model.transformer.layers.1.ffn.expert_sel: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.attn_pre.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.attn_pre.weight: 20.2963
	 Train l2_norm/update/model.transformer.layers.1.attn_pre.weight: 0.0006
	 Train l2_norm/grad/model.transformer.layers.1.attn_pre.weight: 0.0000
	 Train l2_norm/moment/model.transformer.layers.1.ffn_pre.weight: 0.0000
	 Train l2_norm/param/model.transformer.layers.1.ffn_pre.weight: 20.2985
	 Train l2_norm/update/model.transformer.layers.1.ffn_pre.weight: 0.0004
	 Train l2_norm/grad/model.transformer.layers.1.ffn_pre.weight: 0.0000
	 Train l2_norm/moment/model.embedding.weight: 0.0000
	 Train l2_norm/param/model.embedding.weight: 221.7120
	 Train l2_norm/update/model.embedding.weight: 0.0160
	 Train l2_norm/grad/model.embedding.weight: 0.0001
	 Train l2_norm/moment/model.lm_head.weight: 0.0004
	 Train l2_norm/param/model.lm_head.weight: 127.9880
	 Train l2_norm/update/model.lm_head.weight: 0.0775
	 Train l2_norm/grad/model.lm_head.weight: 0.0007
	 Train l2_norm/moment/model.lm_head.bias: 0.0000
	 Train l2_norm/param/model.lm_head.bias: 6.2696
	 Train l2_norm/update/model.lm_head.bias: 0.0065
	 Train l2_norm/grad/model.lm_head.bias: 0.0000
	 Train l2_norm/moment/model.out_norm.weight: 0.0000
	 Train l2_norm/param/model.out_norm.weight: 20.2937
	 Train l2_norm/update/model.out_norm.weight: 0.0008
	 Train l2_norm/grad/model.out_norm.weight: 0.0000
	 Train l2_norm/grad/global: 0.0015
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                        model_inference        38.98%       83.591s        93.96%      201.493s      201.493s       0.000us         0.00%     4328.553s     4328.553s         716 b    -540.01 Gb     886.51 Mb    -275.08 Gb             1
                                            aten::empty         0.05%     117.650ms         0.06%     121.002ms       3.506us       0.000us         0.00%       0.000us       0.000us      40.05 Mb      40.05 Mb     110.53 Gb     110.53 Gb         34516
                                          aten::random_         0.00%      21.600us         0.00%      21.600us      21.600us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
                                             aten::item         0.02%      40.595ms         0.32%     694.763ms      13.078us       0.000us         0.00%      96.508ms       1.817us        -276 b        -256 b           0 b           0 b         53124
                              aten::_local_scalar_dense         0.06%     118.835ms         0.31%     654.168ms      12.314us      96.494ms         0.11%      96.508ms       1.817us         -20 b         -20 b           0 b           0 b         53124
                                           Unrecognized         0.07%     142.057ms         0.07%     142.057ms       1.353ms        8.691s        10.34%        8.691s      82.773ms           0 b           0 b           0 b           0 b           105
                                            aten::zeros         0.01%      15.339ms         0.05%     107.633ms      16.144us       0.000us         0.00%      18.511ms       2.777us         173 b           0 b      34.85 Gb           0 b          6667
                                            aten::zero_         0.07%     152.437ms         1.27%        2.729s      31.477us       0.000us         0.00%        4.443s      51.250us           0 b           0 b           0 b           0 b         86685
                                               aten::to         0.03%      66.073ms        32.29%       69.246s     864.408us       0.000us         0.00%       67.595s     843.796us     135.00 Gb           0 b      69.30 Gb           0 b         80108
                                         aten::_to_copy         0.09%     186.852ms        32.26%       69.180s     973.474us       0.000us         0.00%       67.595s     951.168us     135.00 Gb       2.72 Kb      69.30 Gb           0 b         71065
                                    aten::empty_strided         0.11%     235.521ms         0.11%     238.441ms       3.227us       0.000us         0.00%       0.000us       0.000us     135.00 Gb     135.00 Gb      74.61 Gb      74.61 Gb         73897
                                            aten::copy_         0.18%     395.132ms        32.33%       69.327s     564.914us       67.668s        80.49%       67.669s     551.401us           0 b           0 b           0 b           0 b        122721
                                        cudaMemcpyAsync        31.86%       68.324s        31.86%       68.326s     680.753us       0.000us         0.00%       4.128us       0.000us           0 b           0 b           0 b           0 b        100369
                                        model_inference         0.00%       0.000us         0.00%       0.000us       0.000us      201.427s       239.58%      201.427s      201.427s           0 b           0 b           0 b           0 b             1
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      38.312ms         0.05%      38.312ms       0.877us           0 b           0 b           0 b           0 b         43696
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         0.04%      92.671ms         0.04%      93.056ms       4.653ms       0.000us         0.00%       0.864us       0.043us           0 b           0 b           0 b           0 b            20
                               cudaPointerGetAttributes         0.00%     356.731us         0.00%     394.320us       3.521us       8.832us         0.00%       8.832us       0.079us           8 b           8 b      -1.00 Kb      -1.00 Kb           112
                                          cudaHostAlloc         0.03%      64.908ms         0.04%      88.539ms       2.330ms       8.864us         0.00%      37.986us       1.000us          -8 b          -8 b       1.00 Kb       1.00 Kb            38
                         Memcpy DtoH (Device -> Pinned)         0.00%       0.000us         0.00%       0.000us       0.000us     101.751ms         0.12%     101.751ms       2.018us           0 b           0 b           0 b           0 b         50410
                                  cudaStreamSynchronize         0.39%     844.508ms         0.39%     844.511ms       8.613us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         98054
                                              aten::add         3.04%        6.520s         3.06%        6.572s       1.112ms      21.344ms         0.03%      21.369ms       3.615us     135.00 Gb     135.00 Gb       4.98 Gb       4.98 Gb          5911
                                           aten::detach         0.00%       3.413ms         0.00%       9.885ms       4.085us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2420
                                                 detach         0.00%       6.473ms         0.00%       6.473ms       2.675us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2420
                                            aten::clone         0.01%      13.830ms         0.04%      91.051ms      18.582us       0.000us         0.00%      21.436ms       4.375us          40 b           0 b       9.87 Gb           0 b          4900
                                            aten::split         0.00%     262.123us         0.00%     702.262us      17.557us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            40
                                           aten::narrow         0.00%       4.888ms         0.01%      14.821ms       5.041us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2940
                                            aten::slice         0.05%     109.531ms         0.06%     128.997ms       2.348us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         54932
                                       aten::as_strided         0.06%     129.115ms         0.06%     129.115ms       0.441us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b        293096
                     Optimizer.step#DecoupledAdamW.step         5.78%       12.389s        11.77%       25.233s        1.262s       0.000us         0.00%     4260.880s      213.044s         -40 b    -120.00 Mb       2.65 Gb     -71.51 Gb            20
           Optimizer.zero_grad#DecoupledAdamW.zero_grad         0.00%       2.907ms         0.00%       2.907ms     138.443us       0.000us         0.00%       0.000us       0.000us           0 b           0 b      -5.87 Gb      -5.87 Gb            21
                                       aten::lift_fresh         0.00%       3.887ms         0.00%       3.887ms       0.088us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         44096
                                          aten::detach_         0.00%       3.122ms         0.00%       4.420ms       1.662us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2660
                                                detach_         0.00%       1.298ms         0.00%       1.298ms       0.488us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b          2660
                     Optimizer.step#DecoupledAdamW.step         0.00%       0.000us         0.00%       0.000us       0.000us       25.225s        30.00%       25.225s        1.261s           0 b           0 b           0 b           0 b            20
                         Memcpy HtoD (Pinned -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      98.849us         0.00%      98.849us       1.236us           0 b           0 b           0 b           0 b            80
                                             aten::set_         0.00%       8.255ms         0.00%       8.255ms       2.359us       0.000us         0.00%       0.000us       0.000us        -636 b        -636 b     -69.50 Kb     -69.50 Kb          3500
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       4.009ms         0.00%       4.009ms       1.878us           0 b           0 b           0 b           0 b          2134
                                        aten::new_empty         0.00%       2.582ms         0.00%       8.450ms       4.856us       0.000us         0.00%       0.000us       0.000us         640 b           4 b       1.52 Gb           0 b          1740
                                        aten::embedding        -0.03%  -58837.505us         0.03%      63.775ms       3.189ms       0.000us         0.00%     206.402us      10.320us           0 b           0 b      70.08 Mb           0 b            20
                                          aten::reshape         0.02%      38.414ms         0.06%     130.667ms       4.043us       0.000us         0.00%      14.376ms       0.445us           0 b           0 b       7.18 Gb           0 b         32320
                                             aten::view         0.04%      87.767ms         0.04%      87.767ms       1.472us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         59640
                                     aten::index_select         0.00%     659.660us         0.03%      62.986ms       3.149ms     174.914us         0.00%     186.114us       9.306us           0 b           0 b      70.08 Mb           0 b            20
                                          aten::resize_         0.00%       9.967ms         0.01%      11.316ms       2.526us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       5.74 Gb       5.74 Gb          4480
                                       cudaLaunchKernel         1.28%        2.740s         1.69%        3.614s      12.397us       0.000us         0.00%     349.990ms       1.200us           0 b           0 b           0 b           0 b        291539
                       Runtime Triggered Module Loading         0.20%     436.184ms         0.20%     436.196ms       4.194ms       2.970ms         0.00%       2.973ms      28.582us           0 b           0 b           0 b           0 b           104
                                  Lazy Function Loading         0.01%      14.646ms         0.01%      14.648ms      54.252us       35.201s        41.87%       35.201s     130.376ms           0 b           0 b           0 b           0 b           270
void at::native::(anonymous namespace)::indexSelectL...         0.00%       0.000us         0.00%       0.000us       0.000us     174.914us         0.00%     174.914us       8.746us           0 b           0 b           0 b           0 b            20
                                           aten::select         0.10%     221.185ms         0.12%     267.365ms       2.060us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b        129816
                                               aten::eq         0.01%      13.176ms         0.02%      35.453ms      30.563us       2.261ms         0.00%       2.266ms       1.954us           0 b           0 b     660.00 Kb     659.00 Kb          1160
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.371ms         0.00%       2.371ms       1.943us           0 b           0 b           0 b           0 b          1220
                                    aten::nonzero_numpy         0.00%     838.548us         0.02%      34.230ms      85.576us       0.000us         0.00%       4.890ms      12.225us           0 b           0 b      11.27 Mb           0 b           400
                                          aten::nonzero         0.02%      44.102ms         0.07%     158.142ms      60.824us      24.251ms         0.03%      24.345ms       9.363us           0 b           0 b      40.64 Mb       1.00 Kb          2600
                                     cudaGetDeviceCount         0.00%       0.080us         0.00%       0.080us       0.080us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
                                  cudaFuncGetAttributes         0.00%       3.939ms         0.01%      11.345ms      10.475us       0.000us         0.00%       1.760ms       1.625us           0 b           0 b           0 b           0 b          1083
void at_cuda_detail::cub::DeviceReduceSingleTileKern...         0.00%       0.000us         0.00%       0.000us       0.000us       6.684ms         0.01%       6.684ms       2.571us           0 b           0 b           0 b           0 b          2600
                                    cudaPeekAtLastError         0.00%       1.227ms         0.00%       1.227ms       0.045us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         27120
                                 cudaDeviceGetAttribute         0.00%       5.547ms         0.00%       5.547ms       0.435us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         12762
void at_cuda_detail::cub::DeviceCompactInitKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.110ms         0.00%       4.110ms       1.581us           0 b           0 b           0 b           0 b          2600
void at_cuda_detail::cub::DeviceSelectSweepKernel<at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.745ms         0.01%       6.745ms       2.594us           0 b           0 b           0 b           0 b          2600
                                                aten::t         0.01%      18.193ms         0.02%      48.506ms       4.022us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         12060
                                        aten::transpose         0.02%      41.110ms         0.03%      60.062ms       3.006us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         19980
                                           aten::unbind         0.00%       2.198ms         0.00%       4.944ms      10.300us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b           480
                                             aten::ones         0.00%     761.088us         0.02%      42.556ms     265.973us       0.000us         0.00%       0.000us       0.000us      40.03 Mb         620 b           0 b           0 b           160
                                            aten::fill_         0.11%     245.612ms         1.24%        2.652s      28.769us        4.097s         4.87%        4.445s      48.210us           0 b           0 b           0 b           0 b         92197
                                             aten::tril         0.01%      24.934ms         0.01%      24.948ms     623.705us       0.000us         0.00%       0.000us       0.000us      40.00 Mb      40.00 Mb           0 b           0 b            40
void at::native::vectorized_elementwise_kernel<2, at...         0.00%       0.000us         0.00%       0.000us       0.000us     763.461us         0.00%     763.461us       1.539us           0 b           0 b           0 b           0 b           496
                                              aten::cat         0.01%      21.052ms         0.02%      34.723ms      17.021us       9.071ms         0.01%       9.085ms       4.453us      40.24 Mb      40.24 Mb       2.07 Gb       2.07 Gb          2040
void at::native::(anonymous namespace)::CatArrayBatc...         0.00%       0.000us         0.00%       0.000us       0.000us       1.108ms         0.00%       1.108ms       2.519us           0 b           0 b           0 b           0 b           440
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     648.290us         0.00%     648.290us       1.487us           0 b           0 b           0 b           0 b           436
                                            aten::stack         0.00%       1.525ms         0.00%       8.083ms      50.520us       0.000us         0.00%       0.000us       0.000us      40.24 Mb           0 b           0 b           0 b           160
                                  cudaStreamIsCapturing         0.00%      82.189us         0.00%      82.189us       0.814us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b           101
                                             cudaMalloc         0.01%      15.849ms         0.01%      15.849ms     148.117us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b           107
void at::native::vectorized_elementwise_kernel<2, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.540ms         0.00%       3.540ms       1.394us           0 b           0 b           0 b           0 b          2540
                                              aten::sum         1.22%        2.611s         1.57%        3.360s      73.774us     150.503ms         0.18%     298.950ms       6.564us       2.84 Mb       2.84 Mb     706.49 Mb     523.98 Mb         45541
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us     148.345ms         0.18%     148.345ms       3.552us           0 b           0 b           0 b           0 b         41760
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     135.785ms         0.16%     135.785ms       3.252us           0 b           0 b           0 b           0 b         41760
                                           Buffer Flush         0.01%      16.682ms         0.01%      17.092ms     156.805us       11.380s        13.54%       11.380s     104.403ms         -16 b         -16 b    -200.11 Mb    -200.11 Mb           109
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     147.090ms         0.17%     147.090ms       5.402us           0 b           0 b           0 b           0 b         27227
                                        aten::ones_like         0.00%       1.243ms         0.00%       6.336ms      16.673us       0.000us         0.00%     588.615us       1.549us           0 b           0 b     730.00 Kb           0 b           380
                                       aten::empty_like         0.01%      12.493ms         0.02%      52.339ms       7.916us       0.000us         0.00%       0.000us       0.000us           0 b           0 b      12.85 Gb           0 b          6612
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.077ms         0.00%       2.077ms       2.885us           0 b           0 b           0 b           0 b           720
                                       aten::zeros_like         0.00%     779.461us         0.00%       5.359ms      13.008us       0.000us         0.00%     776.068us       1.884us           0 b           0 b     607.27 Mb           0 b           412
                                            LayerNormFn         0.04%      80.674ms         0.64%        1.364s       1.240ms       0.000us         0.00%       78.320s      71.200ms           0 b           0 b       3.55 Gb      -2.81 Mb          1100
                        flash_attn::layer_norm_fwd_impl         0.44%     935.256ms         0.58%        1.240s       1.127ms      64.657ms         0.08%       78.320s      71.200ms           0 b           0 b       8.59 Mb      -1.50 Gb          1100
                                         cuLaunchKernel         0.21%     442.256ms         0.29%     628.198ms       7.391us       0.000us         0.00%     4160.949s      48.955ms           0 b           0 b           0 b           0 b         84996
                           _layer_norm_fwd_1pass_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      64.657ms         0.08%      64.657ms       6.719us           0 b           0 b           0 b           0 b          9623
                                  cudaDeviceSynchronize         3.93%        8.438s         3.93%        8.438s      21.917ms       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b           385
                                        cudaEventRecord         0.10%     209.476ms         0.10%     220.443ms       1.979us       0.000us         0.00%       39.781s     357.097us           0 b           0 b           0 b           0 b        111400
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us        3.943s         4.69%        3.943s      69.743us           0 b           0 b           0 b           0 b         56534
                                   cudaEventElapsedTime         0.02%      50.541ms         0.02%      50.541ms       0.908us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         55662
                                    Command Buffer Full         1.12%        2.401s         1.12%        2.401s      79.263us     4199.377s      4994.87%     4199.377s     138.648ms           0 b           0 b           0 b           0 b         30288
                                              aten::max         0.01%      20.885ms         0.02%      50.606ms      15.335us      15.243ms         0.02%      15.258ms       4.624us         240 b           4 b       1.58 Mb           0 b          3300
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       9.931ms         0.01%       9.931ms       4.598us           0 b           0 b           0 b           0 b          2160
void at::native::(anonymous namespace)::write_indice...         0.00%       0.000us         0.00%       0.000us       0.000us       1.456ms         0.00%       1.456ms       3.639us           0 b           0 b           0 b           0 b           400
                                         aten::bincount         0.01%      28.222ms         0.11%     244.974ms      97.212us      19.557ms         0.02%      59.546ms      23.630us           0 b     -15.47 Kb       1.93 Mb      -2.46 Mb          2520
                                              aten::min         0.01%      18.896ms         0.02%      45.781ms      17.745us      13.380ms         0.02%      13.397ms       5.193us         240 b          12 b       1.23 Mb           0 b          2580
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       7.843ms         0.01%       7.843ms       5.447us           0 b           0 b           0 b           0 b          1440
                       Memcpy DtoH (Device -> Pageable)         0.00%       0.000us         0.00%       0.000us       0.000us       67.336s        80.09%       67.336s      16.630ms           0 b           0 b           0 b           0 b          4049
void at::cuda::kernelHistogram1D<long, long, long, 1...         0.00%       0.000us         0.00%       0.000us       0.000us      13.279ms         0.02%      13.279ms       9.222us           0 b           0 b           0 b           0 b          1440
                                           aten::cumsum         0.00%       5.546ms         0.00%       9.001ms      25.003us       1.368ms         0.00%       1.375ms       3.819us           0 b           0 b     180.00 Kb     180.00 Kb           360
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 214.451s
Self CUDA time total: 84.074s
