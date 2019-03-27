#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, os, json, random
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SER example')
    parser.add_argument('--input', default='datasets/iemocap/fusion/train.txt', type=str, help='path to list file')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    last_dialog_name = ""
    sentence = []

    for i, line in enumerate(open(args.input, 'r')):
        line = line.strip()
        if line == '':
            continue

        sent_name, wave_file, text, label = line.split('\t')

        if "XX" in sent_name:
            continue

        for line in open(os.path.join("datasets/iemocap/smile/arff", sent_name + ".arff"), 'r'):
            line = line.strip()
            if line.startswith("'unknown'"):
                cols = line.split(',')
                feature = cols[1:385]
                sentence.append((feature, label))
                break

        dialog_name = '_'.join(sent_name.split('_')[:-1])

        if i == 0:
            last_dialog_name = dialog_name
        elif last_dialog_name != dialog_name:
            print("{}\t{}\t{}\t{}".format(
                len(sentence),
                len(feature),
                '\t'.join(['\t'.join(f) for f, _ in sentence]),
                '\t'.join([l for _, l in sentence])
            ))
            sys.stdout.flush()
            sentence = []
            last_dialog_name = dialog_name

    if len(sentence) > 0:
        print("{}\t{}\t{}\t{}".format(
            len(sentence),
            len(feature),
            '\t'.join(['\t'.join(f) for f, _ in sentence]),
            '\t'.join([l for _, l in sentence])
        ))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
