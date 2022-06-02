# Setup

```
conda activate minDPR  # Same libraries
conda deactivate
```

## Toy
```
python train.py /tmp/model data/NQ/train3.json data/NQ/train3.json --max_length 30 --t5_name t5-small --gpus 0 --no_shuffle --batch_size 2 --num_contexts 3  --dropout 0 --num_training_steps 8 --num_warmup_steps 2 --num_report_steps 2 --start_step_val 5 --lr 1e-3
python test.py /tmp/model data/NQ/train3.json --batch_size 1 --num_workers 1 --gpus 0 --num_contexts 3 --max_length 30  --pred /tmp/pred
```

# Training

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py /data/local/minFiD_runs/nq_base/model ../FiD/open_domain_data/NQ/train.json ../FiD/open_domain_data/NQ/dev.json --num_contexts 100 --batch_size 2 --batch_size_val 4 --max_length 250 --lr 0.00005 --dropout 0.1 --num_warmup_steps 1000 --num_training_steps 15000 --start_step_val 10000 --weight_decay 0.01 --clip 1. --num_workers 2 --seed 42 --t5_name t5-base --use_checkpoint --grad_accumulation 4 --num_report_steps 500 --gpus 0,1,2,3,4,5,6,7
```

# Running the public FiD model

```
torchrun --standalone --nnodes=1 --nproc_per_node=3 test.py ../FiD/pretrained_models/nq_reader_base data/NQ/train3.json --batch_size 1 --num_workers 1 --gpus 0,1,2 --num_contexts 100 --pred /tmp/pred
torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py ../FiD/pretrained_models/nq_reader_base ../FiD/open_domain_data/NQ/dev.json --batch_size 12 --num_workers 2 --gpus 0,1,2,3,4,5,6,7 --num_contexts 100 --pred pred_dev.txt  # ~11 minutes on one.cs, 3 minutes on two.cs, EM 49.14 (4304/8757)
torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py ../FiD/pretrained_models/nq_reader_base ../FiD/open_domain_data/NQ/test.json --batch_size 12 --num_workers 2 --gpus 0,1,2,3,4,5,6,7 --num_contexts 100 --pred pred_test.txt  # ~5 minutes on one.cs, 2 minutes on two.cs, EM 50.06 (1808/3610) <-- was 50.03 on one.cs, so GPUs can cause diff
```
