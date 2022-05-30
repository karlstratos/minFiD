import sys

mine = '/common/home/jl2529/repositories/minFiD/pred_dev.txt'
theirs = '/common/home/jl2529/repositories/FiD/checkpoint/nq_dev/final_output.txt'

my_preds = {}
answers = {}
my_scores = {}
with open(mine) as f:
    for line in f:
        i, pred, golds, score = line.strip().split('\t')
        i = int(i)
        my_preds[i] =  pred.strip()
        answers[i] = eval(golds)
        my_scores[i] = float(score)

their_preds = {}
with open(theirs) as f:
    for line in f:
        toks = line.strip().split('\t')
        i = int(toks[0])
        if len(toks) == 1:
            print('WARNING: theirs has an empty line')
            pred = ''
        else:
            pred = toks[1].strip()

        if i in their_preds and their_preds[i] != pred:
            print('WARNING: theirs already had index', i)
            print(their_preds[i], 'vs', pred)

        if pred == '':
            if not i in their_preds:
                print(f'  populating {i} with an empty pred')
            else:
                if their_preds[i] == '':
                    print(f'  repopulating {i}, {their_preds[i]} with an empty pred')
                else:
                    print(f'  rewriting {i}, {their_preds[i]} with an empty pred')

        if i in their_preds and their_preds[i] == '' and pred != '':
            print(their_preds[i], '->', pred)

        their_preds[i] = pred

assert len(my_preds) == len(their_preds), f'{len(my_preds)} vs {len(their_preds)}'
for i in my_preds:
    assert my_preds[i] == their_preds[i], f'{i}: {my_preds[i]} vs {their_preds[i]}'


print(my_preds[1745], 'vs', their_preds[1745])
print(my_preds[7237], 'vs', their_preds[7237])
