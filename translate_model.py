import argparse
import os
import torch
import torch.nn as nn

from model import load_model

def main(args):
    model, saved_args = load_model(args.model, torch.device('cpu'))
    saved_args.max_input_length = saved_args.max_length
    saved_args.base_seq2seq = saved_args.t5_name
    model.seq2seq = model.t5
    delattr(model, "t5")
    torch.save({'sd': model.state_dict(), 'args': saved_args}, args.outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('outpath', type=str)
    args = parser.parse_args()
    main(args)
