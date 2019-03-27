#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""
from typing import Optional, Awaitable

__version__ = '0.0.1'

import sys, time, logging, os, json, random
import numpy as np

np.set_printoptions(precision=20)
logger = logging.getLogger("tornado.application")
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

import tornado.ioloop
import tornado.web
import tornado.websocket
import wave
import uuid
import subprocess


import matplotlib.pyplot as plt


def plot(x, title, save=False):

    plt.figure(figsize=(20, 10))
    plt.title(title)

    ylim = [np.min(x), np.max(x)]
    # ylim = [-1, 1]

    # グラフ左
    plt.subplot(2, 1, 1)
    plt.ylim(ylim)
    plt.plot(range(1, len(x[:, 0]) + 1), x[:, 0], 'b')
    plt.grid()
    plt.title("{} L".format(title))

    # # グラフ右
    # plt.subplot(2, 1, 2)
    # plt.ylim(ylim)
    # plt.plot(range(1, len(x[:, 1]) + 1), x[:, 1], 'b')
    # plt.grid()
    # plt.title("{} R".format(title))

    if save:
        plt.savefig('{}.png'.format(title))

    plt.show()
    plt.close()


SAMPLE_SIZE = 2         # 量子化ビットサイズ 16 bit (2 bytes)
SAMPLE_RATE = 48000     # サンプリングレート 48000 Hz
PATH = './output.wav'


def make_wave_file(voice):

    v = np.array(voice).ravel()
    d = (v * 32767).astype(np.int16)

    # # フレームごとに分割する
    # n_channels = 1
    # framesize = 512
    # samplewidth = SAMPLE_SIZE
    # samplerate = SAMPLE_RATE
    # n_frames = int(d.shape[0] / framesize)
    # frame_data = np.reshape(d[:(framesize * n_frames)], (n_frames, framesize, n_channels))
    # frame_sec = float(framesize) / float(samplerate)
    #
    # logger.info("Channel num: {:d}".format(n_channels))
    # logger.info("Sample width: {:d} bits".format(samplewidth))
    # logger.info("Sampling rate: {:d} Hz".format(samplerate))
    # logger.info("Frame num: {:d}".format(n_frames))
    # logger.info("Frame sec: {:.6f}".format(frame_sec))
    #
    # # フレームごとに平均する
    # frame_avg = np.average(frame_data, axis=1)
    # logger.info("frame min: {}".format(np.min(frame_avg)))
    # logger.info("frame max: {}".format(np.max(frame_avg)))
    # plot(frame_avg, "frame_avg")
    #
    # # フレームごとに最大値を取る
    # frame_avg = np.max(frame_data, axis=1)
    # plot(frame_avg, "frame_max")
    #
    # # フレームごとに最小値を取る
    # frame_avg = np.min(frame_data, axis=1)
    # plot(frame_avg, "frame_min")
    #
    # # dB に変換する
    # # E0 = 2 ** (量子化bit - 1) => 最大値
    # E0 = 10 ** -4  # 最小値 (soundfile では 0~1 に正規化される)
    # dB = 20 * np.log10((np.abs(frame_data) + 10 ** -3) / E0)
    # logger.info("dB min: {}".format(np.min(dB)))
    # logger.info("dB max: {}".format(np.max(dB)))
    #
    # # フレームごとに dB を平均する
    # frame_avg = np.average(dB, axis=1)
    # plot(frame_avg, "dB")

    wave_name = "{}.wav".format(str(uuid.uuid4()))

    with wave.open(wave_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_SIZE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(d.tobytes('C'))

    return wave_name


import chainer
from chainer import cuda
from chainer import initializers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt


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


def emotion(voice):
    X_test = []

    for v in voice:
        wave_name = make_wave_file(v)

        command = "/Data/haradatm/src/opensmile-2.3.0/SMILExtract " \
                  "-C /Data/haradatm/src/opensmile-2.3.0/config/IS09_emotion.conf " \
                  "-noconsoleoutput " \
                  "-I {} " \
                  "-O /dev/stdout".format("{}".format(wave_name))
        # logger.debug(command)

        features = []
        with subprocess.Popen(command.strip().split(' '), stdout=subprocess.PIPE) as proc:
            for line in proc.stdout:
                line = line.decode('utf-8').strip()

                if not line.startswith("@"):
                    if line == "":
                        continue
                    cols = line.split(',')
                    features.append(np.array(cols[1:-1], 'f'))
        os.remove(wave_name)

        X_test.append(np.array(features))

    print('# test X: {}, dim: {}'.format(len(X_test), X_test[0].shape[1]))
    print('# class: {}, labels: {}'.format(len(labels), labels))
    sys.stdout.flush()

    results = []

    for x in X_test:
        results.append(predict(x, args.gpu))

    return results


class WebSocketHandler(tornado.websocket.WebSocketHandler):

    def check_origin(self, origin):
        return True

    def open(self):
        logger.info("open")
        self.voice = []

    def on_message(self, message):
        logger.info("on_message")
        self.voice.append(np.frombuffer(message, dtype='float32'))

        if len(self.voice) > 5:
            self.voice.pop(0)

        results = emotion(self.voice)
        print(results)

    def on_close(self):
        logger.info("on_close")
        self.voice.clear()


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('index.html', sent='')

    def post(self):
        sent = self.get_argument('sent')
        logger.info(sent)
        if sent:
            self.write(json.dumps(dict({'PER': ['Obama'], 'LOC': ['White House']})))


if __name__ == '__main__':
    global model
    global labels
    global index2label

    import argparse
    parser = argparse.ArgumentParser(description='SER example: NStep LSTM')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--layer', type=int, default=2, help='number of layes for turn')
    parser.add_argument('--unit',  type=int, default=300, help='number of units for turn')
    parser.add_argument('--model', default='models/final.model', type=str, help='trained model file (.model)')
    parser.add_argument('--label', default='models/labels.pkl', type=str, help='saved label file (.pkl)')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    labels = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3}
    index2label = {v: k for k, v in labels.items()}

    model = SER(n_layers=args.layer, n_inputs=384, n_outputs=len(labels), n_units=args.unit)
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    elif args.ideep:
        model.to_intel64()

    BASE_DIR = os.path.dirname(__file__)

    application = tornado.web.Application([
        (r'/', MainHandler),
        (r"/ser", WebSocketHandler)
    ],
        template_path=os.path.join(BASE_DIR, 'templates'),
        static_path=os.path.join(BASE_DIR, 'static'),
    )

    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
