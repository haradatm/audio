#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" SER: Speech Emotion Recognition example

[usage] python train_turn-mlp-with_cm.py --batchsize 5 --epoch 100 --layer 1 --unit 200 --dropout 0.4 --train datasets/05-train.txt --test datasets/05-test.txt --out 05-turn-7 2>&1 | tee 05-turn-7-train.log


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
from chainer import initializers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, classification_report


def load_data(filename, labels={}):
    X, y = [], []

    for i, line in enumerate(open(filename, 'r')):
        # if i > 1:
        #     break

        line = line.strip()
        if line == '':
            continue

        row = line.split('\t')
        if len(row) < 3:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        n_turn = int(row[0])
        n_feature = int(row[1])

        X.append(np.asarray(row[2:(n_feature * n_turn)+2], dtype=np.float32).reshape((n_turn, n_feature)))

    print('Loading dataset ... done.')
    sys.stdout.flush()

    return X


# Network definition
class SER(chainer.Chain):

    def __init__(self, n_layers=1, n_inputs=384, n_outputs=4, n_units=300, dropout_rate=0.1, class_weight=None):
        self.dropout_rate = dropout_rate
        self.class_weight = class_weight
        super(SER, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepLSTM(n_layers, n_inputs, n_units, dropout_rate)
            self.l2 = L.Linear(n_units, n_outputs, initialW=initializers.HeNormal())
            # self.af = F.relu
            # self.af = F.leaky_relu

    def __call__(self, xs, ts):
        ys = self.forward(xs)

        accum_loss = None
        accum_accuracy = None

        for i in range(len(ts)):
            y, t = self.l2(ys[i]), ts[i]
            loss, accuracy = F.softmax_cross_entropy(y, t, class_weight=self.class_weight), F.accuracy(y, t)
            accum_loss = loss if accum_loss is None else accum_loss + loss
            accum_accuracy = accuracy if accum_accuracy is None else accum_accuracy + accuracy

        return accum_loss, accum_accuracy

    def forward(self, xs):
        hx, cx = None, None
        hx, cx, ys = self.l1(hx, cx, xs)
        return ys

    def predict(self, xs):
        ys = self.forward(xs)
        return [F.softmax(self.l2(y)) for y in ys]


def to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def predict(x, gpu):
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        X = chainer.Variable(to_device(gpu, x))
        y = model.predict([X])[0]
        y_prob = cuda.to_cpu(y.data)
        y_pred = np.argmax(y_prob, axis=1).tolist()
        return "{" + ', '.join(["'{}': {:.6f}".format(index2label[y], float(p[y])) for y, p in zip(y_pred, y_prob)]) + "}"


def main():
    global model
    global index2label

    import argparse
    parser = argparse.ArgumentParser(description='SER example: NStep LSTM')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--layer', type=int, default=2, help='number of layes for turn')
    parser.add_argument('--unit',  type=int, default=300, help='number of units for turn')
    parser.add_argument('--model', default='models/final.model', type=str, help='trained model file (.model)')
    parser.add_argument('--label', default='models/labels.pkl', type=str, help='saved label file (.pkl)')
    parser.add_argument('--test',  default='datasets/test.txt', type=str, help='testing file (.txt)')
    parser.add_argument('--noplot', action='store_true', help='disable PlotReport extension')
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
        chainer.config.use_cudnn = 'never'

    # データの読み込み
    labels = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3}
    X_test = load_data(args.test, labels=labels)
    n_class = len(labels)

    print('# test X: {}, dim: {}'.format(len(X_test), X_test[0].shape[1]))
    print('# class: {}, labels: {}'.format(n_class, labels))
    sys.stdout.flush()

    model = SER(n_layers=args.layer, n_inputs=X_test[0].shape[1], n_outputs=n_class, n_units=args.unit)
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    elif args.ideep:
        model.to_intel64()

    index2label = {v: k for k, v in labels.items()}

    for x in X_test:
        result = predict(x, args.gpu)
        print(result)


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
