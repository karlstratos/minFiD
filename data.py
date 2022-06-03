import csv
import random
import torch

from file_handling import read_json
from pathlib import Path


class FiDDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.samples = read_json(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], index


def tensorize(samples, num_contexts, tokenizer, max_length, shuffle=True):
    indices = []
    questions = []
    passages_list = []
    targets = []

    for sample, index in samples:
        passages = ['question: {} title: {} context: {}'.format(
            sample['question'], context['title'], context['text'])
                    for context in sample['ctxs'][:num_contexts]]

        if 'target' in sample:
            target = sample['target']
        elif 'answers' in sample:
            target = random.choice(sample['answers']) if shuffle else \
                     sample['answers'][0]
        else:
            target = None

        passages_list.append(passages)
        targets.append(target)
        indices.append(index)

    I = torch.LongTensor(indices)
    P = []
    P_mask = []
    for passages in passages_list:
        # (num_passages, len_passage)
        passages_encoded = tokenizer(passages, padding='max_length',
                                     truncation=True, max_length=max_length,
                                     add_special_tokens=False,  # no </s> needed
                                     return_tensors='pt')
        P.append(passages_encoded['input_ids'].unsqueeze(0))
        P_mask.append(passages_encoded['attention_mask'].unsqueeze(0))
    P = torch.cat(P, dim=0)  # (B, num_passages, len_passage)
    P_mask = torch.cat(P_mask, dim=0).bool()  # (B, num_passages, len_passage)


    targets_encoded = tokenizer(targets, padding='longest', return_tensors='pt')
    T = targets_encoded['input_ids']  # (B, len_target)
    T_mask = targets_encoded['attention_mask'].bool()  # (B,  len_target)

    # https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration.forward.labels
    T = T.masked_fill(~T_mask, -100)

    return I, T, P, P_mask
