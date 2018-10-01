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


import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report


def load_data(filename, labels={}, n_feature=384):
    X, y, z = [], [], []

    for i, line in enumerate(open(filename, 'rU')):
        # if i == 0:
        #     continue

        line = line.strip()
        if line == '':
            continue

        row = line.split('\t')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        n_turn = int(row[0])
        for label in row[-(n_turn+1):]:
            if label not in labels:
                sys.stderr.write('invalid label: {}\n'.format(label))
                sys.exit(1)

        X.append(chainer.Variable(xp.asarray(row[1:-(n_turn+1)], dtype=np.float32).reshape((n_turn, n_feature))))
        y.append(chainer.Variable(xp.asarray([labels[x] for x in row[-(n_turn+1):-1]], dtype=np.int32)))
        z.append(chainer.Variable(xp.asarray([labels[row[-1]]])))

    print('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y, z, labels


# # Network definition
class SER(chainer.Chain):

    def __init__(self, n_input, n_layer, n_units, n_output):
        super(SER, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepBiLSTM(n_layer, n_input, n_units, 0.25)
            self.l2 = L.Linear(n_units * 2, n_output)

    def __call__(self, xs, ts):
        accum_loss = None
        accum_accuracy = None

        hx = None
        cx = None

        hx, cx, ys = self.l1(hx, cx, xs)

        for i in range(len(ts)):
            y, t = self.l2(ys[i]), ts[i]
            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            accum_loss = loss if accum_loss is None else accum_loss + loss
            accum_accuracy = accuracy if accum_accuracy is None else accum_accuracy + accuracy

        return accum_loss, accum_accuracy

    def predict(self, xs):
        hx = None
        cx = None

        hx, cx, ys = self.l1(hx, cx, xs)

        return [F.softmax(self.l2(y)) for y in ys]


def batch_tuple(generator, batch_size):
    batch = []
    for line in generator:
        batch.append(line)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch))
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch))


def main():
    global xp

    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--layer', type=int, default=1,    help='Number of layes for turn')
    parser.add_argument('--unit',  type=int, default=256,  help='Number of units for turn')
    parser.add_argument('--dim',   type=int, default=384,  help='Number of dimensions')
    parser.add_argument('--batchsize', '-b', type=int, default=10,  help='Number of images in each mini-batch')
    parser.add_argument('--test',   default='test.tsv',  type=str, help='evaluating file (.txt)')
    parser.add_argument('--model',  default='final.model', type=str, help='model file (.model)')
    parser.add_argument('--label',  default='label.pkl', type=str, help='label file (.pkl)')
    parser.add_argument('--noplot', dest='plot', action='store_true', help='Disable PlotReport extension')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()
        cuda.cupy.random.seed(seed)

    xp = cuda.cupy if args.gpu >= 0 else np

    # データの読み込み
    labels = pickle.load(open(args.label, 'rb'))
    X_test, y_test, z_test, labels = load_data(args.test, labels=labels)

    n_class = len(labels)
    index2label = {v: k for k, v in labels.items()}

    print('# test X: {}, y: {}, class: {}'.format(len(X_test), len(y_test), n_class))
    sys.stdout.flush()

    model = SER(args.dim, args.layer, args.unit, n_class)
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    test_iter = batch_tuple([(s, t1, t2) for s, t1, t2 in zip(X_test, y_test, z_test)], args.batchsize)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):

        trues = []
        preds = []
        probs = []

        for X, y, z in test_iter:
            y_pred = model.predict(X)

            for pred in y_pred:
                pred = cuda.to_cpu(pred.data)
                idx = np.argmax(pred, axis=0)

                preds.append([index2label[x] for x in idx])
                probs.append([float(p[i]) for i, p in zip(idx, pred)])

                for i, j in zip(preds[-1], probs[-1]):
                    print("{}:{:.4f}".format(i, j), end='\t')
                print()

            for i in range(len(y)):
                trues.append([index2label[x] for x in cuda.to_cpu(y[i].data)])

        print("\n==== Classification report ====\n")
        print(classification_report(
            [inner for outer in trues for inner in outer],
            [inner for outer in preds for inner in outer]
        ))


if __name__ == '__main__':
    main()
