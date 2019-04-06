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

        for label in row[-(n_turn):]:
            if label not in labels:
                labels[label] = len(labels)

        X.append(np.asarray(row[2:-(n_turn)], dtype=np.float32).reshape((n_turn, n_feature)))
        y.append(np.asarray([labels[x] for x in row[-(n_turn):]], dtype=np.int32))

    print('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y, labels


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
    parser = argparse.ArgumentParser(description='SER example: NStep LSTM')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--layer', type=int, default=2, help='number of layes for turn')
    parser.add_argument('--unit',  type=int, default=300, help='number of units for turn')
    parser.add_argument('--batchsize', '-b', type=int, default=8, help='number of data in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.001, help='value of learning rate')
    parser.add_argument('--weightdecay', '-w', type=float, default=0., help='value of exponential decay rate')
    parser.add_argument('--dropout', '-d', type=float, default=0.4, help='dropout rate')
    parser.add_argument('--epoch', '-e', type=int, default=300, help='number of sweeps over the dataset to train')
    parser.add_argument('--train', default='datasets/iemocap/smile/train-dialog.txt', type=str, help='training file (.txt)')
    parser.add_argument('--eval',  default='datasets/iemocap/smile/test-dialog.txt', type=str, help='evaluating file (.txt)')
    parser.add_argument('--out', '-o', default='result-dialog-lstm-1', help='directory to output the result')
    parser.add_argument('--use_classweight', action='store_true', help='use class weight')
    parser.add_argument('--ideep', action='store_true', help='use ideep backend')
    parser.add_argument('--noplot', action='store_true', help='disable PlotReport extension')
    parser.add_argument('--test', action='store_true', help='use tiny datasets for quick tests')
    # parser.set_defaults(use_class_weight=True)
    # parser.set_defaults(test=True)
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

    model_dir = args.out
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # データの読み込み
    labels = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3}
    X_train, y_train, labels = load_data(args.train, labels=labels)
    X_eval,  y_eval,  labels = load_data(args.eval, labels=labels)
    n_class = len(labels)

    y_train_list = [flatten for inner in y_train for flatten in inner]
    y_eval_list  = [flatten for inner in y_eval  for flatten in inner]

    print('# train X: {}, y: {}, dim: {}, counts: {}'.format(len(X_train), len(y_train), X_train[0].shape[1], [y_train_list.count(x) for x in sorted(labels.values())]))
    print('# eval  X: {}, y: {}, dim: {}, counts: {}'.format(len(X_eval),  len(y_eval),  X_eval[0].shape[1],  [y_eval_list.count(x)  for x in sorted(labels.values())]))
    print('# class: {}, labels: {}'.format(n_class, labels))

    class_weight = None
    if args.use_classweight:
        class_count = np.array([y_train_list.count(x) for x in sorted(labels.values())], 'f')
        class_weight = np.sum(class_count) / class_count
    print('# class weight: {}'.format(class_weight))
    sys.stdout.flush()

    if args.gpu >= 0:
        class_weight = cuda.to_gpu(class_weight)

    with open(os.path.join(args.out, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    model = SER(n_layers=args.layer, n_inputs=X_train[0].shape[1], n_outputs=n_class, n_units=args.unit, dropout_rate=args.dropout, class_weight=class_weight)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    elif args.ideep:
        model.to_intel64()

    # 学習率
    lr = args.learnrate

    # # 減衰率
    # momentum = args.momentum

    # 重み減衰
    weight_decay = args.weightdecay

    # # 勾配上限
    # grad_clip = 3

    # Setup optimizer
    optimizer = chainer.optimizers.Adam(alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08, eta=1.0, weight_decay_rate=weight_decay)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

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
            X = [chainer.Variable(to_device(args.gpu, x)) for x in X]
            t = [chainer.Variable(to_device(args.gpu, y)) for y in t]

            # 勾配を初期化
            model.cleargrads()

            # 順伝播させて誤差と精度を算出
            loss, accuracy = model(X, t)
            sum_train_loss += float(loss.data)
            sum_train_accuracy1 += float(accuracy.data)
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
                X = [chainer.Variable(to_device(args.gpu, x)) for x in X]
                t = [chainer.Variable(to_device(args.gpu, y)) for y in t]

                # 順伝播させて誤差と精度を算出
                loss, accuracy = model(X, t)
                sum_test_loss += float(loss.data)
                sum_test_accuracy1 += float(accuracy.data)
                sum_test_accuracy2 += .0
                K += len(t)

                y = model.predict(X)
                for pred, true in zip(y, t):
                    y_pred += np.argmax(cuda.to_cpu(pred.data), axis=1).tolist()
                    y_true += cuda.to_cpu(true.data).tolist()

        cm = confusion_matrix(y_true, y_pred)
        cm2 = np.apply_along_axis(lambda x: x / sum(x), 1, cm)
        uar = np.nanmean(np.diag(cm2))
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
            optimizer.lr
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
            X = [chainer.Variable(to_device(args.gpu, x)) for x in X]
            y = model.predict(X)
            for pred, true in zip(y, t):
                y_pred += np.argmax(cuda.to_cpu(pred.data), axis=1).tolist()
                y_true += true.tolist()

        print("\n==== Confusion matrix 1 (early_stopped-loss) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (early_stopped-loss) ====\n")
        cm2 = np.apply_along_axis(lambda x: x / sum(x), 1, cm)
        uar = np.nanmean(np.diag(cm2))

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
            plt.title('Confusion matrix: UAR = {:.6f}'.format(uar))
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
            X = [chainer.Variable(to_device(args.gpu, x)) for x in X]
            y = model.predict(X)
            for pred, true in zip(y, t):
                y_pred += np.argmax(cuda.to_cpu(pred.data), axis=1).tolist()
                y_true += true.tolist()

        print("\n==== Confusion matrix 1 (early_stopped-uar) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (early_stopped-uar) ====\n")
        cm2 = np.apply_along_axis(lambda x: x / sum(x), 1, cm)
        uar = np.nanmean(np.diag(cm2))

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
            plt.title('Confusion matrix: UAR = {:.6f}'.format(uar))
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
            X = [chainer.Variable(to_device(args.gpu, x)) for x in X]
            y = model.predict(X)
            for pred, true in zip(y, t):
                y_pred += np.argmax(cuda.to_cpu(pred.data), axis=1).tolist()
                y_true += true.tolist()

        print("\n==== Confusion matrix 1 (final model) ====\n")
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))

        print("\n==== Confusion matrix 2 (final model) ====\n")
        cm2 = np.apply_along_axis(lambda x: x / sum(x), 1, cm)
        uar = np.nanmean(np.diag(cm2))

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
            plt.title('Confusion matrix: UAR = {:.6f}'.format(uar))
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
