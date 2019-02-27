#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, os, json, random, re
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
# handler = logging.FileHandler(filename="log.txt")
handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    logger.info(pp.pformat(obj))


start_time = time.time()


import chainer
import chainer.functions as F
from datasets.bertlib.modeling import BertConfig, BertModel
from datasets.bertlib.tokenization import FullTokenizer


class BertTask(chainer.Chain):
    def __init__(self, model):
        super(BertTask, self).__init__()
        with self.init_scope():
            self.bert = model


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SER example')
    parser.add_argument('--input', default='datasets/iemocap/fusion/train.txt', type=str, help='path to list file')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    vocab_file = "models/uncased_L-12_H-768_A-12/vocab.txt"
    bert_config_file = "models/uncased_L-12_H-768_A-12/bert_config.json"
    init_checkpoint = "models/uncased_L-12_H-768_A-12/arrays_bert_model.ckpt.npz"

    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    bert_config = BertConfig.from_json_file(bert_config_file)
    bert = BertModel(config=bert_config)
    extractor = BertTask(bert)
    with np.load(init_checkpoint) as f:
        d = chainer.serializers.NpzDeserializer(f, path='', strict=True)
        d.load(extractor)

    for i, line in enumerate(open(args.input, 'r')):
        line = line.strip()
        if line == '':
            continue

        sent, wave_file, text, label = line.split('\t')

        if "XX" in sent:
            continue

        text = text.replace('. . .', '…')
        text = re.sub('\[.*?\]', "", text)

        # if text == '':
        #     continue

        tokens_a = tokenizer.tokenize(text)
        tokens = ["[CLS]"]
        segment_ids = [0]

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        max_position_embeddings = bert_config.max_position_embeddings
        x1 = np.array([input_ids[:max_position_embeddings]], 'i')
        x2 = np.array([input_mask[:max_position_embeddings]], 'f')
        x3 = np.array([segment_ids[:max_position_embeddings]], 'i')
        feature = bert.get_pooled_output(x1, x2, x3).data[0]

        print('\t'.join(["%.6f" % float(x) for x in feature]) + '\t' + label)
        sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
