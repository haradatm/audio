#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" SER: Speech Emotion Recognition example (3-D)

This is a implementation for the paper "3-D Convolutional Recurrent Neural Networks with Attention Model for Speech Emotion Recognition" by Mingyi Chen, Xuanji He, Jing Yang, and Han Zhang.

[usage] python train_ser.py --batchsize 10 --epoch 100 --layer 1 --unit 200 --train datasets/fbank/01-train.txt --test datasets/fbank/01-test.txt --out 01-turn 2>&1 | tee 01-turn.log


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
import wave
import python_speech_features as ps
import bz2


def load_and_mfcc(filename, labels={}, train=False):
    X, y = [], []

    for i, line in enumerate(open(filename, 'r')):
        # if i >= 200:
        #     break

        line = line.strip()
        if line == '':
            continue
        if line.startswith('#'):
            continue

        row = line.split('\t')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        wave_file, emotion = row

        if emotion not in labels:
            labels[emotion] = len(labels)

        with wave.open(os.path.join(root_dir, wave_file), 'rb') as file:
            params = file.getparams()
            nchannels, sampwidth, framerate, nframes, comptipe, compname = params[:6]
            str_data = file.readframes(nframes)
            wavedata = np.fromstring(str_data, dtype=np.short)

        winlen = 0.025
        winstep = 0.01
        nfft = int(winlen * framerate)
        nfilt = 40
        numcep = 13

        mfcc_feat = ps.mfcc(wavedata, samplerate=framerate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft)
        delta1 = ps.delta(mfcc_feat, 2)
        delta2 = ps.delta(delta1, 2)
        time = mfcc_feat.shape[0]

        # max_frames = 1024
        max_frames = 300

        if train and time > max_frames:
            if emotion in ['ang', 'neu', 'sad']:
                for j in range(2):
                    if j == 0:
                        begin = 0
                        end = begin + 300
                    else:
                        begin = time - 300
                        end = time
                    part = mfcc_feat[begin:end, :]
                    delta11 = delta1[begin:end, :]
                    delta21 = delta2[begin:end, :]
                    feature = np.empty((len(part), numcep, 3), dtype=np.float32)
                    feature[:, :, 0] = part
                    feature[:, :, 1] = delta11
                    feature[:, :, 2] = delta21
                    X.append(feature)
                    y.append(np.array([labels[emotion]], 'i'))
            else:
                frames = divmod(time, 100)[0] + 1
                for i in range(frames):
                    begin = 100 * i
                    end = begin + 300
                    part = mfcc_feat[begin:end, :]
                    delta11 = delta1[begin:end, :]
                    delta21 = delta2[begin:end, :]
                    feature = np.empty((len(part), numcep, 3), dtype=np.float32)
                    feature[:, :, 0] = part
                    feature[:, :, 1] = delta11
                    feature[:, :, 2] = delta21
                    X.append(feature)
                    y.append(np.array([labels[emotion]], 'i'))

        else:
            feature = np.empty((time, numcep, 3), dtype=np.float32)
            feature[:, :, 0] = mfcc_feat
            feature[:, :, 1] = delta1
            feature[:, :, 2] = delta2
            X.append(feature)
            y.append(np.array([labels[emotion]], 'i'))

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y, labels


def load_and_fbank(filename, labels={}, train=False):
    X, y = [], []

    for i, line in enumerate(open(filename, 'r')):
        # if i >= 200:
        #     break

        line = line.strip()
        if line == '':
            continue
        if line.startswith('#'):
            continue

        row = line.split('\t')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        wave_file, emotion = row

        if emotion not in labels:
            labels[emotion] = len(labels)

        with wave.open(os.path.join(root_dir, wave_file), 'rb') as file:
            params = file.getparams()
            nchannels, sampwidth, framerate, nframes, comptipe, compname = params[:6]
            str_data = file.readframes(nframes)
            wavedata = np.fromstring(str_data, dtype=np.short)

        winlen = 0.025
        winstep = 0.01
        nfft = int(winlen * framerate)
        nfilt = 40

        mel_spec = ps.logfbank(wavedata, samplerate=framerate, winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft)
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)
        time = mel_spec.shape[0]

        # max_frames = 1024
        max_frames = 300

        if train and time > max_frames:
            if emotion in ['ang', 'neu', 'sad']:
                for j in range(2):
                    if j == 0:
                        begin = 0
                        end = begin + 300
                    else:
                        begin = time - 300
                        end = time
                    part = mel_spec[begin:end, :]
                    delta11 = delta1[begin:end, :]
                    delta21 = delta2[begin:end, :]
                    feature = np.empty((len(part), nfilt, 3), dtype=np.float32)
                    feature[:, :, 0] = part
                    feature[:, :, 1] = delta11
                    feature[:, :, 2] = delta21
                    X.append(feature)
                    y.append(np.array([labels[emotion]], 'i'))
            else:
                frames = divmod(time, 100)[0] + 1
                for i in range(frames):
                    begin = 100 * i
                    end = begin + 300
                    part = mel_spec[begin:end, :]
                    delta11 = delta1[begin:end, :]
                    delta21 = delta2[begin:end, :]
                    feature = np.empty((len(part), nfilt, 3), dtype=np.float32)
                    feature[:, :, 0] = part
                    feature[:, :, 1] = delta11
                    feature[:, :, 2] = delta21
                    X.append(feature)
                    y.append(np.array([labels[emotion]], 'i'))

        else:
            feature = np.empty((time, nfilt, 3), dtype=np.float32)
            feature[:, :, 0] = mel_spec
            feature[:, :, 1] = delta1
            feature[:, :, 2] = delta2
            X.append(feature)
            y.append(np.array([labels[emotion]], 'i'))

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X, y, labels


class SER(chainer.Chain):

    def __init__(self, n_layers=1, n_inputs=3, n_outputs=3, linear_num=786, cell_num=128, atten1=64, atten2=1, hidden1=64, dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        super(SER, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(n_inputs, 128, ksize=(5, 3), stride=(1, 1), pad=(2, 1))
            self.c2 = L.Convolution2D(128, 512, ksize=(5, 3), stride=(1, 1), pad=(2, 1))
            self.b1 = L.BatchNormalization(128)
            self.b2 = L.BatchNormalization(512)
            self.cl = L.Linear(None, linear_num)
            self.lstm = L.NStepBiLSTM(n_layers, linear_num, cell_num, dropout_rate)
            self.a1 = L.Linear(cell_num * 2, atten1)
            self.a2 = L.Linear(atten1, atten2)
            self.l3 = L.Linear(cell_num * 2, hidden1)
            self.b3 = L.BatchNormalization(hidden1)
            self.l4 = L.Linear(hidden1, n_outputs)
            self.af = F.leaky_relu

    def __call__(self, xs, ts):
        y = self.forward(xs)
        t = F.concat(ts, axis=0)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(self, xs):
        y = self.forward(xs)
        return F.softmax(y)

    def forward(self, xs):
        h = chainer.dataset.convert.concat_examples(xs, padding=-1)
        h = h.transpose(0, 3, 1, 2)                             # (1, 3, 300, 40)

        h = self.c1(h)                                          # (1, 128, 300, 40)
        h = self.b1(h)
        h = self.af(h)
        h = F.max_pooling_2d(h, ksize=(2, 4), stride=(2, 4))    # (1, 128, 150, 10)
        # h = F.dropout(h, ratio=self.dropout_rate)

        h = self.c2(h)                                          # (1, 512, 150, 10)
        h = self.b2(h)
        h = self.af(h)
        h = F.max_pooling_2d(h, ksize=(1, 2), stride=(1, 2))    # (1, 512, 150, 5)
        # h = F.dropout(h, ratio=self.dropout_rate)

        h = h.transpose(0, 2, 1, 3)                             # (1, 150, 512, 5)
        h = h.reshape(len(xs) * h.shape[1], -1)                 # (1 * 150, 6144)
        h = self.cl(h)                                          # (1 * 150, 786)
        h = h.reshape(len(xs), -1, 786)                         # (1, 150, 786)

        last_h, last_c, ys = self.lstm(None, None, [_ for _ in h])
        y_len = [len(y) for y in ys]
        y_section = np.cumsum(y_len[:-1])
        ay = self.a2(F.relu(self.a1(F.dropout(F.concat(ys, axis=0), ratio=self.dropout_rate))))
        ays = F.split_axis(ay, y_section, 0)

        h_list = []
        for y, ay in zip(ys, ays):
            h_list.append(F.sum(y * F.broadcast_to(F.softmax(ay, axis=0), y.shape), axis=0)[None, :])
        h = F.concat(h_list, axis=0)
        # h = F.dropout(h, ratio=self.dropout_rate)

        h = self.l3(h)
        h = self.b3(h)
        h = self.af(h)

        y = self.l4(h)

        return y


from sklearn.utils import shuffle as skshuffle


def batch_iter(data, batch_size, shuffle=False, random_state=123):
    batch = []
    shuffled_data = np.copy(data)

    if shuffle:
        shuffled_data = skshuffle(shuffled_data, random_state=random_state)

    for line in shuffled_data:
        batch.append(line)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch))
            # yield batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch))
        # yield batch


def sorted_batch_iter(data, batch_size, shuffle=False, random_state=123):
    batch = []
    sorted_data = sorted(data, key=lambda x: len(x[0]), reverse=True)

    for line in sorted_data:
        batch.append(line)
        if len(batch) == batch_size:
            batch = skshuffle(batch, random_state=random_state)
            yield tuple(list(x) for x in zip(*batch))
            # yield batch
            batch = []
    if batch:
        batch = skshuffle(batch, random_state=random_state)
        yield tuple(list(x) for x in zip(*batch))
        # yield batch


def to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def save_as_pickled_object(obj, filepath):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    n_bytes = sys.getsizeof(bytes_out)
    with bz2.BZ2File(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def main():
    global root_dir

    import argparse
    parser = argparse.ArgumentParser(description='SER example: 3-D ACRNN')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--layer', type=int, default=1, help='number of layes for turn')
    parser.add_argument('--batchsize', '-b', type=int, default=40, help='number of images in each mini-batch')
    parser.add_argument('--dropout', '-d', type=float, default=0.1, help='value of dropout rate')
    parser.add_argument('--epoch', '-e', type=int, default=300, help='number of sweeps over the dataset to train')
    parser.add_argument('--train', default='datasets/01-train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--valid', default='datasets/01-valid.txt', type=str, help='validating file (.txt)')
    parser.add_argument('--test',  default='datasets/01-test.txt', type=str, help='evaluating file (.txt)')
    parser.add_argument('--out', '-o', default='iemocap-fbank-21-300_b040_e300_d010_adam', help='directory to output the result')
    parser.add_argument('--noplot', dest='noplot', action='store_true', help='disable PlotReport extension')
    parser.add_argument('--optim',  default='adam', choices=['adam', 'adadelta'], help='type of optimizer')
    parser.add_argument('--type',  default='fbank', choices=['fbank', 'mfcc'], help='type of feature')
    parser.add_argument('--resume', default='', type=str, help='path to resume models')
    parser.add_argument('--start_epoch', default=1, type=int, help='epoch number at start')
    parser.add_argument('--datasets_rootdir', default='', type=str, help='path to datasets')
    parser.add_argument('--datasets_only', action='store_true', help='make and save datasets only')
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

    root_dir = args.datasets_rootdir

    # データの読み込み
    labels = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3}

    if args.type == 'fbank':
        X_train, y_train, labels = load_and_fbank(args.train, labels=labels, train=True)
        X_valid, y_valid, labels = load_and_fbank(args.valid, labels=labels)
        X_test,  y_test,  labels = load_and_fbank(args.test,  labels=labels)
    elif args.type == 'mfcc':
        X_train, y_train, labels = load_and_mfcc(args.train, labels=labels, train=True)
        X_valid, y_valid, labels = load_and_mfcc(args.valid, labels=labels)
        X_test,  y_test,  labels = load_and_mfcc(args.test,  labels=labels)
    else:
        raise ValueError("Only support fbank or mfcc.")

    X_train_section = np.cumsum([len(x) for x in X_train][:-1])
    X_train = np.concatenate(X_train, axis=0)

    X_valid_section = np.cumsum([len(x) for x in X_valid][:-1])
    X_valid = np.concatenate(X_valid, axis=0)

    eps = 1e-5
    tensor = np.concatenate((X_train, X_valid), axis=0)
    mean = np.mean(tensor, axis=0)
    std = np.std(tensor, axis=0)
    with open(os.path.join(args.out, 'mean-std.pkl'), 'wb') as f:
        pickle.dump((mean, std), f)

    X_train = (X_train - mean) / (std + eps)
    X_train = np.split(X_train, X_train_section, 0)

    X_valid = (X_valid - mean) / (std + eps)
    X_valid = np.split(X_valid, X_valid_section, 0)

    with open(os.path.join(args.out, "labels.pkl"), 'wb') as f:
        pickle.dump(labels, f)
    with bz2.BZ2File(os.path.join(args.out, "data.pkl.bz2"), 'wb') as f:
        pickle.dump((X_train, y_train, X_valid, y_valid, X_test, y_test), f)

    with bz2.BZ2File(os.path.join(args.out, "data.pkl.bz2"), 'rb') as f:
        (X_train, y_train, X_valid, y_valid, X_test, y_test) = pickle.load(f)
    with open(os.path.join(args.out, "labels.pkl"), 'rb') as f:
        labels = pickle.load(f)

    n_class = len(labels)

    print('# train X: {}, y: {}'.format(len(X_train), len(y_train)))
    print('# valid X: {}, y: {}'.format(len(X_valid), len(y_valid)))
    print('# test  X: {}, y: {}'.format(len(X_test),  len(y_test)))
    print('# class: {}'.format(n_class))
    print(labels)
    sys.stdout.flush()

    if args.datasets_only:
        return

    model = SER(n_layers=args.layer, n_inputs=3, n_outputs=n_class, dropout_rate=args.dropout)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # 学習率
    lr = 0.001

    # 学習率の減衰
    lr_decay = 0.99

    # 重み減衰
    decay = 0.0001

    # 勾配上限
    grad_clip = 3

    # Setup optimizer (Optimizer の設定)
    if args.optim == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=lr, beta1=0.9, beta2=0.999, weight_decay_rate=decay, eps=1e-8)
    elif args.optim == 'adadelta':
        optimizer = chainer.optimizers.AdaDelta()
    else:
        raise ValueError("Only support adam or adadelta.")

    # optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
    optimizer.setup(model)

    if args.resume:
        print('resuming model and state at: {}'.format(args.start_epoch))
        if args.gpu >= 0: model.to_cpu()
        chainer.serializers.load_npz(os.path.join(args.resume, 'final.model'), model)
        chainer.serializers.load_npz(os.path.join(args.resume, 'final.state'), optimizer)
        if args.gpu >= 0: model.to_gpu()
        sys.stdout.flush()

    # プロット用に実行結果を保存する
    train_loss = []
    train_accuracy1 = []
    train_accuracy2 = []
    test_loss = []
    test_accuracy1 = []
    test_accuracy2 = []

    min_loss = float('inf')
    min_epoch = 0

    start_epoch = args.start_epoch

    start_at = time.time()
    cur_at = start_at

    # Learning loop
    for epoch in range(start_epoch, args.epoch + 1):

        # training
        train_iter = batch_iter([(x, t) for x, t in zip(X_train, y_train)], args.batchsize, shuffle=True, random_state=seed)
        sum_train_loss = 0.
        sum_train_accuracy1 = 0.
        sum_train_accuracy2 = 0.
        K = 0

        for X, t in train_iter:
            x = to_device(args.gpu, X)
            t = to_device(args.gpu, t)

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
        valid_iter = batch_iter([(x, t) for x, t in zip(X_valid, y_valid)], 1, shuffle=False)
        sum_test_loss = 0.
        sum_test_accuracy1 = 0.
        sum_test_accuracy2 = 0.
        K = 0

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for X, t in valid_iter:
                x = to_device(args.gpu, X)
                t = to_device(args.gpu, t)

                # 順伝播させて誤差と精度を算出
                loss, accuracy = model(x, t)
                sum_test_loss += float(loss.data) * len(t)
                sum_test_accuracy1 += float(accuracy.data) * len(t)
                sum_test_accuracy2 += .0
                K += len(t)

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
                    'V/loss={:.6f} '
                    'V/acc1={:.6f} '
                    'V/acc2={:.6f} '
                    'V/sec= {:.6f} '
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
            min_epoch = epoch
            print('saving early-stopped model at epoch {}'.format(min_epoch))
            chainer.serializers.save_npz(os.path.join(model_dir, 'early_stopped.model'), model)
            chainer.serializers.save_npz(os.path.join(model_dir, 'early_stopped.state'), optimizer)
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
            # ylim2 = [min(train_accuracy1 + test_accuracy2), max(train_accuracy1 + test_accuracy2)]
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
            plt.legend(['train acc1', 'train acc2'], loc="upper right")
            plt.title('Loss and accuracy of train.')

            # グラフ右
            plt.subplot(1, 2, 2)
            plt.ylim(ylim1)
            plt.plot(range(1, len(test_loss) + 1), test_loss, color='C1', marker='x')
            # plt.grid()
            # plt.ylabel('loss')
            plt.legend(['valid loss'], loc="lower left")
            plt.twinx()
            plt.ylim(ylim2)
            plt.plot(range(1, len(test_accuracy1) + 1), test_accuracy1, color='C0', marker='x')
            # plt.plot(range(1, len(test_accuracy2) + 1), test_accuracy2, color='C2', marker='x')
            plt.yticks(np.arange(ylim2[0], ylim2[1], .1))
            plt.grid(True)
            plt.ylabel('accuracy')
            plt.legend(['valid acc1', 'valid acc2'], loc="upper right")
            plt.title('Loss and accuracy of valid.')

            plt.savefig('{}.png'.format(args.out))
            # plt.savefig('{}-train.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        cur_at = now

    # test (early_stopped.model)
    chainer.serializers.load_npz(os.path.join(model_dir, 'early_stopped.model'), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    from sklearn.metrics import confusion_matrix, classification_report
    test_iter = batch_iter([(x, t) for x, t in zip(X_test, y_test)], 1, shuffle=False)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        y_true = []
        y_pred = []

        for X, t in test_iter:
            x = to_device(args.gpu, np.asarray(X, 'f'))
            y = model.predict(x)
            y_pred += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
            y_true += [int(_) for _ in t]

        print("\n==== Confusion matrix (early-stopped model) ====\n")
        index2label = {v: k for k, v in labels.items()}
        sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda x: x[1], reverse=False)]
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))
        sys.stdout.flush()

        # グラフ描画
        if not args.noplot:
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, "{}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = np.arange(len(sorted_labels))
            plt.xticks(tick_marks, sorted_labels, rotation=45)
            plt.yticks(tick_marks, sorted_labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig('{}-cm-early_stopped.png'.format(args.out))
            # plt.savefig('{}-cm-early_stopped.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report (early-stopped model) ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()

    # test (final.model)
    chainer.serializers.load_npz(os.path.join(model_dir, 'final.model'), model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    from sklearn.metrics import confusion_matrix, classification_report
    test_iter = batch_iter([(x, t) for x, t in zip(X_test, y_test)], 1, shuffle=False)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        y_true = []
        y_pred = []

        for X, t in test_iter:
            x = to_device(args.gpu, np.asarray(X, 'f'))
            y = model.predict(x)
            y_pred += np.argmax(cuda.to_cpu(y.data), axis=1).tolist()
            y_true += [int(_) for _ in t]

        print("\n==== Confusion matrix (final model) ====\n")
        index2label = {v: k for k, v in labels.items()}
        sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda x: x[1], reverse=False)]
        cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

        print("\t{}".format("\t".join(sorted_labels)))
        for label, counts in zip(sorted_labels, cm):
            print("{}\t{}".format(label, "\t".join(map(str, counts))))
        sys.stdout.flush()

        # グラフ描画
        if not args.noplot:
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, "{}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
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
