# Setup

Using the same libraries as in minDPR (transformers 4.17.0, torch 1.10.1+cu113)
```
conda activate minDPR 
conda deactivate
```

# Implementation Notes 

- The [original FiD](https://github.com/facebookresearch/FiD) trains for `X` steps saving every `Y`-th step. They [confound](https://github.com/facebookresearch/FiD/blob/25ed1ff0fe0288b80fb5e9e5de8d6346b94b8d48/train_reader.py#L49) steps with batches processed, requiring the user to change `X` and `Y` when using gradient accumulation. This can be fixed by disentangling these values, but I found epoch-based training to be just as effective and simpler. So use `train_epoch.py` instead of `train.py`.
- [Gradient checkpointing](https://arxiv.org/pdf/1604.06174.pdf) ([tutorial](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb)) seems to be now supported in PyTorch's T5 implementation ([here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1020)), so we can probably simplify the code by removing the custom `CheckpointWrapper`. But I think it's simple enough and a useful exercise.
- `transformers` changed the implementation of the [decoder's `T5Block`](https://github.com/huggingface/transformers/blob/56b83cf049823ed074a655eceb28f31e2077c6eb/src/transformers/models/t5/modeling_t5.py#L627) in this version (4.17.0) so that the cross attention layer no longer uses relative attention bias (vs original 3.0.2). I modified the T5 implementation back for replication studies. After replication, I reverted to the newer version, and it seems to work as well or slightly better.  
- Setting `num_workers` too high (or even when it's just 2) sometimes gave me `"ERROR: Unexpected segmentation fault encountered in worker"`. But now it works fine with 2. 
- Training is very computation heavy, but fortunately it worked right away. It was possibly also fortunate that I found out the learning rate for `t5-base` was slightly different from that for `t5-large` which is the released hyperparameter setting (via [this issue](https://github.com/facebookresearch/FiD/issues/11)).  

# Experiments

[Google Sheet](https://docs.google.com/spreadsheets/d/1sxZWNqmFD9Get0WPo_Xt7lxHEbcH3nMXu4TU1DIH4i0/edit#gid=1966517841)

# Commands

## Toy
```
python train_epoch.py /tmp/model data/NQ/train3.json data/NQ/train3.json --max_length 30 --t5_name t5-small --gpus 0 --no_shuffle --batch_size 2 --num_contexts 3  --dropout 0 --epochs 10 --num_warmup_steps 2  --lr 1e-3
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_epoch.py /tmp/model data/NQ/train3.json data/NQ/train3.json --max_length 30 --t5_name t5-small --gpus 0,1,2 --no_shuffle --batch_size 1 --num_contexts 3  --dropout 0 --epochs 10 --num_warmup_steps 2  --lr 1e-3
```

## NQ

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_epoch.py /data/local/minFiD_runs/nq_base_epoch_fix2_newtransformers/model ../FiD/open_domain_data/NQ/train.json ../FiD/open_domain_data/NQ/dev.json --num_contexts 100 --batch_size 2 --batch_size_val 4 --max_length 250 --lr 1e-4 --dropout 0.1 --num_warmup_steps 1000 --epochs 15 --weight_decay 0.01 --clip 1. --num_workers 3 --seed 42 --t5_name t5-base --use_checkpoint --grad_accumulation 4 --gpus 0,1,2,3,4,5,6,7 --data_test ../FiD/open_domain_data/NQ/test.json
```

## TriviaQA

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_epoch.py /data/local/minFiD_runs/tqa_base_epoch_fix2_newtransformers/model ../FiD/open_domain_data/TQA/train.json ../FiD/open_domain_data/TQA/dev.json --num_contexts 100 --batch_size 2 --batch_size_val 4 --max_length 250 --lr 1e-4 --dropout 0.1 --num_warmup_steps 1000 --epochs 15 --weight_decay 0.01 --clip 1. --num_workers 3 --seed 42 --t5_name t5-base --use_checkpoint --grad_accumulation 4 --gpus 0,1,2,3,4,5,6,7 --data_test ../FiD/open_domain_data/TQA/test.json
```

## Running the public FiD model

```
torchrun --standalone --nnodes=1 --nproc_per_node=3 test.py ../FiD/pretrained_models/nq_reader_base data/NQ/train3.json --batch_size 1 --num_workers 1 --gpus 0,1,2 --num_contexts 100 --pred /tmp/pred
torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py ../FiD/pretrained_models/nq_reader_base ../FiD/open_domain_data/NQ/dev.json --batch_size 12 --num_workers 2 --gpus 0,1,2,3,4,5,6,7 --num_contexts 100 --pred pred_dev.txt  # ~11 minutes on one.cs, 3 minutes on two.cs, EM 49.14 (4304/8757)
torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py ../FiD/pretrained_models/nq_reader_base ../FiD/open_domain_data/NQ/test.json --batch_size 12 --num_workers 2 --gpus 0,1,2,3,4,5,6,7 --num_contexts 100 --pred pred_test.txt  # ~5 minutes on one.cs, 2 minutes on two.cs, EM 50.06 (1808/3610) <-- was 50.03 on one.cs, so GPUs can cause diff
```
