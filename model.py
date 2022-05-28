import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist

from evaluate import exact_match_score
from model_ref import FiDT5_ref
from transformers import T5Config, T5ForConditionalGeneration


class FiDT5(nn.Module):

    def __init__(self, t5_name=None, dropout=0.1, saved_model=None):
        super().__init__()
        assert t5_name is not None or saved_model is not None
        if t5_name is not None:
            config = T5Config.from_pretrained(t5_name, dropout_rate=dropout)
            self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name,
                                                                 config=config)
        if saved_model is not None:
            if os.path.isdir(saved_model):
                # Loading reference FiDT5 saved with save_pretrained, must
                # unwrap the encoder and copy.
                self.t5 = T5ForConditionalGeneration.from_pretrained(
                    saved_model)
                t5_ref = FiDT5_ref.from_pretrained(saved_model)
                t5_ref.unwrap_encoder()
                self.t5.encoder = copy.deepcopy(t5_ref.encoder)
            else:  # Loading my model
                pass

        self.wrap_encoder()

    def wrap_encoder(self):
        self.t5.encoder = EncoderWrapper(self.t5.encoder)

    def unwrap_encoder(self):
        self.t5.encoder = self.t5.encoder.encoder
        self.t5.encoder.block = nn.ModuleList([wrap.t5block for wrap in
                                               self.t5.encoder.block])

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.t5.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        for wrap in self.t5.encoder.encoder.block:
            wrap.use_checkpoint = use_checkpoint

    def forward(self, input_ids, attention_mask, labels=None, generate=False,
                max_length=50):
        batch_size, num_passages, len_passage = input_ids.shape

        self.t5.encoder.num_passages = num_passages
        input_ids = input_ids.view(batch_size, -1)
        attention_mask = attention_mask.view(batch_size, -1)

        if generate:
            return self.t5.generate(input_ids, max_length=max_length,
                                    attention_mask=attention_mask)
        else:
            return self.t5(input_ids, attention_mask, labels=labels)


class EncoderWrapper(nn.Module):

    def __init__(self, encoder):
        super().__init__()

        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        # TODO: Try turning this on instead.
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L856
        encoder.block = nn.ModuleList([CheckpointWrapper(t5block) for t5block in
                                       encoder.block])
        self.encoder = encoder

    # This will be called when T5 calls its encoder's forward.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        assert hasattr(self, 'num_passages')
        B, flat = input_ids.shape  # flat = num_p * p_len
        input_ids = input_ids.view(B * self.num_passages, -1)
        attention_mask = attention_mask.view(B * self.num_passages, -1)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)

        # (B * num_p, p_len, dim) -> (B, num_p * p_len, dim)
        outputs.last_hidden_state = outputs.last_hidden_state.view(B, flat, -1)

        return outputs


class CheckpointWrapper(torch.nn.Module):

    def __init__(self, t5block):
        super().__init__()
        self.t5block = t5block
        self.use_checkpoint = False  # Externally set

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.t5block(*inputs, **kwargs)
                empty = torch.empty(0, dtype=torch.float,
                                    device=output[0].device, requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            # Either (1) passing *args after custom_forward or (2) omitting
            # position bias breaks checkpoint (repeats until out of memory).
            output = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, position_bias)
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.t5block(hidden_states, attention_mask, position_bias,
                                  **kwargs)

        return output


def get_mean_em(model, loader, tokenizer, rank=-1, world_size=-1, device=None):
    model.eval()
    if device is None:
        device = torch.device('cpu')

    # We must ensure the wrapped encoder has main_input_name ('input_ids')
    # because of this:
    # https://github.com/huggingface/transformers/blob/518bd02c9b71291333ef374f055a4d1ac3042654/src/transformers/generation_utils.py#L414
    # This seems to be a bad design decision since the mixin class is
    # forecasting properties of its children.
    model.t5.encoder.main_input_name = model.t5.encoder.encoder.main_input_name

    num_examples = 0
    scores = []
    with torch.no_grad():
        for batch in loader:
            I, T, P, P_mask = [tensor.to(device) for tensor in batch]
            outputs = model(P, P_mask, generate=True, max_length=50)

            for j, output in enumerate(outputs):
                pred = tokenizer.decode(output, skip_special_tokens=True)
                golds = loader.dataset.samples[I[j]]['answers']
                score = max([exact_match_score(pred, gold) for gold in golds])
                num_examples += 1
                scores.append(score)

    if world_size != -1:
        sum_em = torch.tensor([sum(scores)], device=device)
        sum_num_examples = torch.tensor([num_examples], device=device)
        dist.reduce(sum_em, 0, op=dist.ReduceOp.SUM)
        dist.reduce(sum_num_examples, 0, op=dist.ReduceOp.SUM)
        mean_em = (sum_em / sum_num_examples).item()
    else:
        mean_em = np.mean(scores)

    return mean_em * 100.
