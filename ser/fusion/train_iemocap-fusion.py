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
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, classification_report


def load_data(filename, labels={}):
    X, y = [], []

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

        feature = [float(x) for x in row[0:-1]]
        label = row[-1]
        if label not in labels:
            labels[label] = len(labels)

        X.append(np.array(feature))           # feature
        y.append(labels[label])     # class

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y, labels


# Network definition
class SER(chainer.Chain):

    def __init__(self, n_layers=1, n_outputs=4, n_units=200, dropout_rate=0.1, class_weight=None):
        self.dropout_rate = dropout_rate
        self.class_weight = class_weight
        super(SER, self).__init__()
        with self.init_scope():
            self.l11 = L.Linear(None, n_units)
            self.b11 = L.BatchNormalization(n_units)
            self.l12 = L.Linear(None, n_units)
            self.b12 = L.BatchNormalization(n_units)
            self.l13 = L.Linear(None, n_units)
            self.b13 = L.BatchNormalization(n_units)

            self.l21 = L.Linear(None, n_units)
            self.b21 = L.BatchNormalization(n_units)
            self.l22 = L.Linear(None, n_units)
            self.b22 = L.BatchNormalization(n_units)
            self.l23 = L.Linear(None, n_units)
            self.b23 = L.BatchNormalization(n_units)

            self.l4 = L.Linear(n_units * 2, n_units)
            self.b4 = L.BatchNormalization(n_units)
            self.l5 = L.Linear(None, n_outputs)

            self.af = F.relu
            # self.af = F.leaky_relu

    def forward(self, x):
        (x1, x2) = F.split_axis(x, [384], axis=1)

        h1 = self.l11(x1)
        h1 = self.b11(h1)
        h1 = self.af(h1)
        h1 = F.dropout(h1, ratio=self.dropout_rate)
        h1 = self.l12(h1)
        h1 = self.b12(h1)
        h1 = self.af(h1)
        h1 = F.dropout(h1, ratio=self.dropout_rate)
        h1 = self.l13(h1)
        h1 = self.b13(h1)
        h1 = self.af(h1)

        h2 = self.l21(x2)
        h2 = self.b21(h2)
        h2 = self.af(h2)
        h2 = F.dropout(h2, ratio=self.dropout_rate)
        h2 = self.l22(h2)
        h2 = self.b22(h2)
        h2 = self.af(h2)
        h2 = F.dropout(h2, ratio=self.dropout_rate)
        h2 = self.l23(h2)
        h2 = self.b23(h2)
        h2 = self.af(h2)

        h = F.concat((h1, h2), axis=1)
        h = F.dropout(h, ratio=self.dropout_rate)
        h = self.l4(h)
        h = self.b4(h)
        h = F.dropout(h, ratio=self.dropout_rate)
        h = self.l5(h)
        return h

    def __call__(self, x, t):
        y = self.forward(x)
        loss = F.softmax_cross_entropy(y, t, class_weight=self.class_weight)
        accuracy = F.accuracy(y, t)
        return loss, accuracy

    def predict(self, x):
        y = self.forward(x)
        return F.softmax(y)


from sklearn.utils import shuffle as skshuffle


def batch_iter(data, batch_size, shuffle=True):
    batch = []
    shuffled_data = np.copy(data)
    if shuffle:
        shuffled_data = skshuffle(shuffled_data)

    for line in shuffled_data:
        batch.append(line)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch))
            # yield batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch))
        # yield batch


def to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SER example: MLP')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--layer', type=int, default=1, help='number of layes for turn')
    parser.add_argument('--unit',  type=int, default=500, help='number of units for turn')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='number of images in each mini-batch')
    parser.add_argument('--dropout', '-d', type=float, default=0.4, help='value of dropout rate')
    parser.add_argument('--weightdecay', default=0.001, type=float, help='value of weight decay rate')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='number of sweeps over the dataset to train')
    parser.add_argument('--train', default='datasets/signate/smile/train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--valid', default='datasets/signate/smile/valid.txt', type=str, help='validation file (.txt)')
    parser.add_argument('--out', '-o', default='result-mlp_cw', help='directory to output the result')
    parser.add_argument('--noplot', action='store_true', help='disable PlotReport extension')
    parser.add_argument('--optim',  default='adam', choices=['adam', 'adadelta'], help='type of optimizer')
    parser.add_argument('--cw', default='none', choices=['none', 'sum', 'norm'], help='type of class weight')
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

    model_dir = args.out
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # データの読み込み
    labels = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3}
    # labels = {'MA_CH': 0, 'FE_AD': 1, 'MA_AD': 2, 'FE_EL': 3, 'FE_CH': 4, 'MA_EL': 5}

    X_train, y_train, labels = load_data(args.train, labels=labels)
    X_eval,  y_eval,  labels = load_data(args.valid, labels=labels)

    n_class = len(labels)

    print('# train X: {}, y: {}, dim: {}, counts: {}'.format(len(X_train), len(y_train), len(X_train[0]), [y_train.count(x) for x in sorted(labels.values())]))
    print('# eval  X: {}, y: {}, dim: {}, counts: {}'.format(len(X_eval),  len(y_eval),  1,               [y_eval.count(x)  for x in sorted(labels.values())]))
    print('# class: {}, labels: {}'.format(n_class, labels))

    class_weight = None
    class_count = np.array([y_train.count(x) for x in sorted(labels.values())], 'f')
    if args.cw == 'sum':
        class_weight = np.sum(class_count) / class_count
    elif args.cw == 'norm':
        class_weight = np.sum(class_count) / class_count
        class_weight = class_weight / np.max(class_weight)
    print('# class_weight: {}'.format(class_weight))
    sys.stdout.flush()

    if args.gpu >= 0:
        class_weight = cuda.to_gpu(class_weight)

    with open(os.path.join(args.out, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    model = SER(n_layers=args.layer, n_outputs=n_class, n_units=args.unit, dropout_rate=args.dropout, class_weight=class_weight)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # 学習率
    lr = 0.001

    # 学習率の減衰
    lr_decay = 0.99

    # 勾配上限
    grad_clip = 3

    # Setup optimizer (Optimizer の設定)
    if args.optim == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=lr, beta1=0.9, beta2=0.999, weight_decay_rate=args.weightdecay, eps=1e-8)
    elif args.optim == 'adadelta':
        optimizer = chainer.optimizers.AdaDelta()
    else:
        raise ValueError("Only support adam or adadelta.")

    # optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    optimizer.setup(model)

    # プロット用に実行結果を保存する
    train_loss = []
    train_accuracy1 = []
    train_accuracy2 = []
    test_loss = []
    test_accuracy1 = []
    test_accuracy2 = []

    min_loss = float('inf')
    best_accuracy = .0

    start_at = time.time()
    cur_at = start_at

    # Learning loop
    for epoch in range(1, args.epoch + 1):

        # training
        train_iter = batch_iter([(x, t) for x, t in zip(X_train, y_train)], args.batchsize, shuffle=True)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        for X, t in train_iter:
            x = to_device(args.gpu, np.asarray(X, 'f'))
            t = to_device(args.gpu, np.asarray(t, 'i'))

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, accuracy = model(x, t)
            sum_train_loss += float(loss.data) * len(t)
            sum_train_accuracy1 += float(accuracy.data) * len(t)
            sum_train_accuracy2 += .0
            K += len(t)

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

        # 訓練データの誤差と,正解精度を表示
        mean_train_loss = sum_train_loss / K
        mean_train_accuracy1 = sum_train_accuracy1 / K
        mean_train_accuracy2 = sum_train_accuracy2 / K
        train_loss.append(mean_train_loss)
        train_accuracy1.append(mean_train_accuracy1)
        train_accuracy2.append(mean_train_accuracy2)
        now = time.time()
        train_throughput = now - cur_at
        cur_at = now

        # evaluation
        test_iter = batch_iter([(x, t) for x, t in zip(X_eval, y_eval)], args.batchsize, shuffle=False)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            y_true = []
            y_pred = []
            for X, t in test_iter:
                x = to_device(args.gpu, np.asarray(X, 'f'))
                t = to_device(args.gpu, np.asarray(t, 'i'))

                # 順伝播させて誤差と精度を算出
                loss, accuracy = model(x, t)
                sum_test_loss += float(loss.data) * len(t)
                sum_test_accuracy1 += float(accuracy.data) * len(t)
                sum_test_accuracy2 += .0
                K += len(t)

                y = model.predict(x)
                y_pred += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
                y_true += t.tolist()

        cm = confusion_matrix(y_true, y_pred)
        cm2 = cm / np.sum(cm, axis=1)
        uar = np.mean(np.diag(cm2))
        sum_test_accuracy2 += uar * K

        # テストデータでの誤差と正解精度を表示
        mean_test_loss = sum_test_loss / K
        mean_test_accuracy1 = sum_test_accuracy1 / K
        mean_test_accuracy2 = sum_test_accuracy2 / K
        test_loss.append(mean_test_loss)
        test_accuracy1.append(mean_test_accuracy1)
        test_accuracy2.append(mean_test_accuracy2)
        now = time.time()
        test_throughput = now - cur_at
        cur_at = now

        logger.info(''
                    '[{:>3d}] '
                    'T/loss={:.6f} '
                    'T/acc1={:.6f} '
                    'T/acc2={:.6f} '
                    'T/sec= {:.6f} '
                    'D/loss={:.6f} '
                    'D/acc1={:.6f} '
                    'D/acc2={:.6f} '
                    'D/sec= {:.6f} '
                    'rate={:.6f}'
                    ''.format(
            epoch,
            mean_train_loss,
            mean_train_accuracy1,
            mean_train_accuracy2,
            train_throughput,
            mean_test_loss,
            mean_test_accuracy1,
            mean_test_accuracy2,
            test_throughput,
            optimizer.rho if args.optim == 'adadelta' else optimizer.lr
        )
        )
        sys.stdout.flush()

        # model と optimizer を保存する
        if args.gpu >= 0: model.to_cpu()
        if mean_test_loss < min_loss:
            min_loss = mean_test_loss
            print('saving early-stopped model (loss) at epoch {}'.format(epoch))
            chainer.serializers.save_npz(os.path.join(model_dir, 'early_stopped-loss.model'), model)
        if mean_test_accuracy2 > best_accuracy:
            best_accuracy = mean_test_accuracy2
            print('saving early-stopped model (uar) at epoch {}'.format(epoch))
            chainer.serializers.save_npz(os.path.join(model_dir, 'early_stopped-uar.model'), model)
        # print('saving final model at epoch {}'.format(epoch))
        chainer.serializers.save_npz(os.path.join(model_dir, 'final.model'), model)
        chainer.serializers.save_npz(os.path.join(model_dir, 'final.state'), optimizer)
        if args.gpu >= 0: model.to_gpu()
        sys.stdout.flush()

        # if args.optim == 'adam':
        #     optimizer.alpha *= lr_decay

        # 精度と誤差をグラフ描画
        if not args.noplot:
            ylim1 = [min(train_loss + test_loss), max(train_loss + test_loss)]
            # ylim2 = [min(train_accuracy1 + test_accuracy1 + train_accuracy2 + test_accuracy2), max(train_accuracy1 + test_accuracy1 + train_accuracy2 + test_accuracy2)]
            ylim2 = [0., 1.]

            # グラフ左
            plt.figure(figsize=(10, 10))

            plt.subplot(1, 2, 1)
            plt.ylim(ylim1)
            plt.plot(range(1, len(train_loss) + 1), train_loss, color='C1', marker='x')
            # plt.grid()
            plt.ylabel('loss')
            plt.legend(['train loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(train_accuracy1) + 1), train_accuracy1, color='C0', marker='x')
            # plt.plot(range(1, len(train_accuracy2) + 1), train_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            # plt.ylabel('accuracy')
            plt.legend(['train acc'], loc="upper right")
            plt.title('Loss and accuracy of train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, color='C1', marker='x')
            # plt.grid()
            # plt.ylabel('loss')
            plt.legend(['test loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, color='C0', marker='x')
            plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            plt.ylabel('accuracy')
            plt.legend(['test acc', 'test uar'], loc="upper right")
            plt.title('Loss and accuracy of test.')

            plt.savefig('{}.png'.format(args.out))
            # plt.savefig('{}-train.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        cur_at = now

    index2label = {v: k for k, v in labels.items()}
    sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda x: x[1], reverse=False)]

    # test (early_stopped model by loss)
    chainer.serializers.load_npz(os.path.join(model_dir, 'early_stopped-loss.model'), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    test_iter = batch_iter([(x, t) for x, t in zip(X_eval, y_eval)], args.batchsize, shuffle=False)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        y_true = []
        y_pred = []

        for X, t in test_iter:
            x = to_device(args.gpu, np.asarray(X, 'f'))
            y = model.predict(x)
            y_pred += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
            y_true += t

        print("\n==== Confusion matrix 1 (early_stopped-loss) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (early_stopped-loss) ====\n")
        cm2 = cm / np.sum(cm, axis=1).reshape(4, 1)
        uar = np.mean(np.diag(cm2))

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm2):
            print("{}\t{}".format(label, "\t".join(map(lambda x: "%.2f" % x, counts))))

        print("\nUAR = {:.6f}".format(float(uar)))
        sys.stdout.flush()

        # グラフ描画
        if not args.noplot:
            plt.figure()
            plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(cm2.shape[0]):
                for j in range(cm2.shape[1]):
                    plt.text(j, i, "{:.2f}".format(cm2[i, j]), horizontalalignment="center", color="white" if cm2[i, j] > cm2.max() / 2 else "black")
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = np.arange(len(sorted_labels))
            plt.xticks(tick_marks, sorted_labels, rotation=45)
            plt.yticks(tick_marks, sorted_labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig('{}-cm-early_stopped-loss.png'.format(args.out))
            # plt.savefig('{}-train_cm.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report (early_stopped-loss) ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()

    # test (early_stopped model by uar)
    chainer.serializers.load_npz(os.path.join(model_dir, 'early_stopped-uar.model'), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    test_iter = batch_iter([(x, t) for x, t in zip(X_eval, y_eval)], args.batchsize, shuffle=False)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        y_true = []
        y_pred = []

        for X, t in test_iter:
            x = to_device(args.gpu, np.asarray(X, 'f'))
            y = model.predict(x)
            y_pred += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
            y_true += t

        print("\n==== Confusion matrix 1 (early_stopped-uar) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (early_stopped-uar) ====\n")
        cm2 = cm / np.sum(cm, axis=1).reshape(4, 1)
        uar = np.mean(np.diag(cm2))

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm2):
            print("{}\t{}".format(label, "\t".join(map(lambda x: "%.2f" % x, counts))))

        print("\nUAR = {:.6f}".format(float(uar)))
        sys.stdout.flush()

        # グラフ描画
        if not args.noplot:
            plt.figure()
            plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(cm2.shape[0]):
                for j in range(cm2.shape[1]):
                    plt.text(j, i, "{:.2f}".format(cm2[i, j]), horizontalalignment="center", color="white" if cm2[i, j] > cm2.max() / 2 else "black")
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = np.arange(len(sorted_labels))
            plt.xticks(tick_marks, sorted_labels, rotation=45)
            plt.yticks(tick_marks, sorted_labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig('{}-cm-early_stopped-uar.png'.format(args.out))
            # plt.savefig('{}-train_cm.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report (early_stopped-uar) ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()

    # test (final model)
    chainer.serializers.load_npz(os.path.join(model_dir, 'final.model'), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    test_iter = batch_iter([(x, t) for x, t in zip(X_eval, y_eval)], args.batchsize, shuffle=False)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        y_true = []
        y_pred = []

        for X, t in test_iter:
            x = to_device(args.gpu, np.asarray(X, 'f'))
            y = model.predict(x)
            y_pred += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
            y_true += t

        print("\n==== Confusion matrix 1 (final model) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (final model) ====\n")
        cm2 = cm / np.sum(cm, axis=1).reshape(4, 1)
        uar = np.mean(np.diag(cm2))

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm2):
            print("{}\t{}".format(label, "\t".join(map(lambda x: "%.2f" % x, counts))))

        print("\nUAR = {:.6f}".format(float(uar)))
        sys.stdout.flush()

        # グラフ描画
        if not args.noplot:
            plt.figure()
            plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(cm2.shape[0]):
                for j in range(cm2.shape[1]):
                    plt.text(j, i, "{:.2f}".format(cm2[i, j]), horizontalalignment="center", color="white" if cm2[i, j] > cm2.max() / 2 else "black")
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = np.arange(len(sorted_labels))
            plt.xticks(tick_marks, sorted_labels, rotation=45)
            plt.yticks(tick_marks, sorted_labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig('{}-cm-final.png'.format(args.out))
            # plt.savefig('{}-cm-final.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report (final model) ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
