# python check_data.py ../FiD/open_domain_data/NQ/train.json
import argparse
import random
import statistics as stat

from file_handling import read_json


def flatten(xss):
    return [x for xs in xss for x in xs]


def print_stats(values, prefix=''):
    value_mean = stat.mean(values)
    value_stdev = stat.stdev(values)
    value_min = min(values)
    value_max = max(values)
    print(prefix +
          f' ({len(values)}): ' +
          f'mean {value_mean:5.2f} [{value_min}, {value_max}], ' +
          f'stdev {value_stdev:5.2f}')

def print_example(x):
    print('-'*80)
    print(f"{x['question']}")
    print(f"{x['answers']}")
    print(f"{x['ctxs'][0]['title']}")
    print(f"{x['ctxs'][0]['text']}")


def main(args):
    random.seed(args.seed)
    examples = read_json(args.data, verbose=True)
    len_question_list = [len(x['question'].split()) for x in examples]
    num_answers_list = [len(x['answers']) for x in examples]
    len_answer_list = [[len(a.split()) for a in x['answers']] for x in examples]
    len_answer_list = flatten(len_answer_list)
    len_context_text_list = [[len(c['text'].split()) for c in x['ctxs']]
                             for x in examples]
    len_context_text_list = flatten(len_context_text_list)
    len_context_title_list = [[len(c['title'].split()) for c in x['ctxs']]
                             for x in examples]
    len_context_title_list = flatten(len_context_title_list)

    print_stats(len_question_list, 'question length')
    print_stats(num_answers_list, 'num answers')
    print_stats(len_answer_list, 'answer length')
    print_stats(len_context_text_list, 'context text length')
    print_stats(len_context_title_list, 'context title length')

    picks = random.sample(examples, args.K)
    for x in picks:
        print_example(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)
