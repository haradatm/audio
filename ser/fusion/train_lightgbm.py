#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Kaggole: Sentiment Analysis on Movie Reviews
    https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
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


import pickle
import wave
import python_speech_features as ps
import bz2
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
# import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt


def load_train_data(filename, labels={}):
    X, y = [], []

    for i, line in enumerate(open(filename, 'r')):
        # if i > 500:
        #     break

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


def load_test_data(filename):
    X, y = [], []

    for i, line in enumerate(open(filename, 'r')):
        # if i == 0:
        #     continue

        line = line.strip()
        if line == '':
            continue

        row = line.split('\t')
        if len(row) < 2:
            sys.stderr.write('invalid record: {}\n'.format(line))
            continue

        X.append(np.array(row))

    logger.info('Loading dataset ... done.')
    sys.stdout.flush()

    return X


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='LightGBM')
    parser.add_argument('--train', default='datasets/signate/smile/train.txt', type=str, help='training file (.txt)')
    parser.add_argument('--test',  default='datasets/signate/smile/test.txt',  type=str, help='testing file (.txt)')
    parser.add_argument('--out', '-o', default='result-lightgbm', help='directory to output the result')
    parser.add_argument('--noplot', dest='noplot', action='store_true', help='disable PlotReport extension')
    parser.add_argument('--cw', dest='cw', action='store_true', help='use class weight')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_dir = args.out
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # データの読み込み
    labels = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3}
    # labels = {'MA_CH': 0, 'FE_AD': 1, 'MA_AD': 2, 'FE_EL': 3, 'FE_CH': 4, 'MA_EL': 5}
    X, Y, labels = load_train_data(args.train, labels=labels)
    n_class = len(labels)

    print('# train X: {}, y: {}, dim: {}, counts: {}'.format(len(X), len(Y), len(X[0]), [Y.count(x) for x in sorted(labels.values())]))
    print('# class: {}, labels: {}'.format(n_class, labels))

    class_weight = None
    if args.cw:
        class_weight = {}
        for k, v in sorted(labels.items(), key=lambda x: x[1]):
            class_weight[v] = len(Y) / Y.count(v)
    print('# class_weight: {}'.format(class_weight))
    print('')

    sys.stdout.flush()

    # classifier = xgb.XGBClassifier()
    classifier = lgb.LGBMClassifier(class_weight=class_weight)

    pipeline = Pipeline([('clf', classifier)])

    parameters = {
        'clf__max_depth': [-1, 5, 10, 20],
        'clf__n_estimators': [50, 100, 200],
    }

    print('# Tuning hyper-parameters for %s' % 'accuracy')
    print('')

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    clf = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, cv=cv, verbose=1, scoring='accuracy')

    clf.fit(X, Y)
    print('Best parameters set found on development set:')
    print('')
    print(clf.best_params_)
    print('')
    print('Grid scores on development set:')
    print('')
    sys.stdout.flush()

    means  = clf.cv_results_['mean_test_score']
    stds   = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, param))
    print('')

    print('Detailed classification report:')
    print('')
    print('The model is trained on the full development set.')
    print('The scores are computed on the full evaluation set.')
    print('')
    sys.stdout.flush()

    clf = clf.best_estimator_
    clf.fit(X, Y)

    with open(os.path.join(args.out, 'model.pkl'), 'wb') as f:
        pickle.dump(clf, f)

    with open(os.path.join(args.out, 'model.pkl'), 'rb') as f:
        clf = pickle.load(f)

    # テストデータの読み込み
    X_test, y_test, labels = load_train_data(args.test, labels=labels)

    print('# test X: {}'.format(len(X_test)))
    sys.stdout.flush()

    y_true, y_pred = y_test, clf.predict(X_test)

    index2label = {v: k for k, v in labels.items()}
    sorted_labels = [k for k, _ in sorted(labels.items(), key=lambda x: x[1], reverse=False)]

    print("\n==== Confusion matrix 1 (test) ====\n")
    cm = confusion_matrix([index2label[x] for x in y_true], [index2label[x] for x in y_pred], labels=sorted_labels)

    print("\t{}".format("\t".join(sorted_labels)))
    for label, counts in zip(sorted_labels, cm):
        print("{}\t{}".format(label, "\t".join(map(str, counts))))
    sys.stdout.flush()

    print("\n==== Confusion matrix 2 (test) ====\n")
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
        plt.savefig('{}-cm-test.png'.format(args.out))
        # plt.savefig('{}-test_cm.png'.format(os.path.splitext(os.path.basename(__file__))[0]))
        # plt.show()
        plt.close()

    print("\n==== Classification report (test) ====\n")
    print(classification_report(
        [sorted_labels[x] for x in y_true],
        [sorted_labels[x] for x in y_pred]
    ))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
    print('time spent: ', time.time() - start_time)
