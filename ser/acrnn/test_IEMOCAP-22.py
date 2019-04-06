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
    parser.add_argument('--batchsize', '-b', type=int, default=1, help='number of images in each mini-batch')
    parser.add_argument('--model', default='models/final.model', type=str, help='trained model file (.model)')
    parser.add_argument('--label', default='models/labels.pkl', type=str, help='saved label file (.pkl)')
    parser.add_argument('--mean',  default='models/mean-std.pkl', type=str, help='saved label file (.pkl)')
    parser.add_argument('--test',  default='datasets/test.txt', type=str, help='testing file (.txt)')
    parser.add_argument('--out', '-o', default='result-17', help='directory to output the result')
    parser.add_argument('--noplot', dest='noplot', action='store_true', help='disable PlotReport extension')
    parser.add_argument('--type',  default='mfcc', choices=['fbank', 'mfcc'], help='type of feature')
    parser.add_argument('--datasets_rootdir', default='', type=str, help='path to datasets')
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

    root_dir = args.datasets_rootdir

    # test
    with open(args.label, 'rb') as f:
        labels = pickle.load(f)
    n_class = len(labels)

    # データの読み込み
    if args.type == 'fbank':
        X_test, y_test, _ = load_and_fbank(args.test, labels=labels)
    elif args.type == 'mfcc':
        X_test, y_test, _ = load_and_mfcc(args.test, labels=labels)
    else:
        raise ValueError("Only support fbank or mfcc.")

    print('# test X: {}'.format(len(X_test)))
    sys.stdout.flush()

    eps = 1e-5
    with open(args.mean, 'rb') as f:
        (mean, std) = pickle.load(f)
    X_test = [(x - mean) / (std + eps) for x in X_test]

    model = SER(n_layers=args.layer, n_inputs=3, n_outputs=n_class)

    chainer.serializers.load_npz(args.model, model)
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

        print("\n==== Confusion matrix ====\n")
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
            plt.savefig('{}-cm.png'.format(args.out))
            # plt.savefig('{}-cm.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
            # plt.show()
            plt.close()

        print("\n==== Classification report ====\n")
        print(classification_report(
            [sorted_labels[x] for x in y_true],
            [sorted_labels[x] for x in y_pred]
        ))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
