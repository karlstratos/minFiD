# Setup

```
conda activate minDPR  # Same libraries
conda deactivate
```

## Toy
```
python train.py /tmp/model data/NQ/train3.json data/NQ/train3.json --max_length 30 --t5_name t5-small --gpus 0 --no_shuffle --batch_size 2 --num_contexts 3  --dropout 0 --num_training_steps 8 --num_warmup_steps 2 --num_report_steps 2 --start_step_val 5 --lr 1e-3

python test.py /tmp/model data/NQ/train3.json --batch_size 1 --num_workers 1 --gpus 0 --num_contexts 3 --max_length 30  --pred /tmp/pred

python train_epoch.py /tmp/model data/NQ/train3.json data/NQ/train3.json --max_length 30 --t5_name t5-small --gpus 0 --no_shuffle --batch_size 2 --num_contexts 3  --dropout 0 --epochs 10 --num_warmup_steps 2  --lr 1e-3
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_epoch.py /tmp/model data/NQ/train3.json data/NQ/train3.json --max_length 30 --t5_name t5-small --gpus 0,1,2 --no_shuffle --batch_size 1 --num_contexts 3  --dropout 0 --epochs 10 --num_warmup_steps 2  --lr 1e-3
```

# Training

```
# This threw "ERROR: Unexpected segmentation fault encountered in worker" with num_workers=2, that's why it was run with num_workers=0. Took 22 hours, about 40 minutes for each report period and 4 minutes for each validation.
# Best val EM 48.90 obtained at the last step check 15000/15000 (beginning of epoch 12). Step 10000/15000 corresponds to epoch 8 (EM 48.09).
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py /data/local/minFiD_runs/nq_base/model ../FiD/open_domain_data/NQ/train.json ../FiD/open_domain_data/NQ/dev.json --num_contexts 100 --batch_size 2 --batch_size_val 4 --max_length 250 --lr 0.00005 --dropout 0.1 --num_warmup_steps 1000 --num_training_steps 15000 --start_step_val 10000 --weight_decay 0.01 --clip 1. --num_workers 0 --seed 42 --t5_name t5-base --use_checkpoint --grad_accumulation 4 --num_report_steps 500 --gpus 0,1,2,3,4,5,6,7

torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py /data/local/minFiD_runs/nq_base/model ../FiD/open_domain_data/NQ/test.json --batch_size 12 --num_workers 2 --gpus 0,1,2,3,4,5,6,7 --num_contexts 100 --max_length 250  --pred /data/local/minFiD_runs/nq_base/pred_test  # 49.53

# Epoch version: more natural to model the relationship between effective batch size and the number of steps
# I'm also trying a diff lr because of this: https://github.com/facebookresearch/FiD/issues/11
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_epoch.py /data/local/minFiD_runs/nq_base_epoch/model ../FiD/open_domain_data/NQ/train.json ../FiD/open_domain_data/NQ/dev.json --num_contexts 100 --batch_size 2 --batch_size_val 4 --max_length 250 --lr 1e-4 --dropout 0.1 --num_warmup_steps 1000 --epochs 15 --weight_decay 0.01 --clip 1. --num_workers 3 --seed 42 --t5_name t5-base --use_checkpoint --grad_accumulation 4 --gpus 0,1,2,3,4,5,6,7
```

# Running the public FiD model

```
torchrun --standalone --nnodes=1 --nproc_per_node=3 test.py ../FiD/pretrained_models/nq_reader_base data/NQ/train3.json --batch_size 1 --num_workers 1 --gpus 0,1,2 --num_contexts 100 --pred /tmp/pred
torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py ../FiD/pretrained_models/nq_reader_base ../FiD/open_domain_data/NQ/dev.json --batch_size 12 --num_workers 2 --gpus 0,1,2,3,4,5,6,7 --num_contexts 100 --pred pred_dev.txt  # ~11 minutes on one.cs, 3 minutes on two.cs, EM 49.14 (4304/8757)
torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py ../FiD/pretrained_models/nq_reader_base ../FiD/open_domain_data/NQ/test.json --batch_size 12 --num_workers 2 --gpus 0,1,2,3,4,5,6,7 --num_contexts 100 --pred pred_test.txt  # ~5 minutes on one.cs, 2 minutes on two.cs, EM 50.06 (1808/3610) <-- was 50.03 on one.cs, so GPUs can cause diff
```
