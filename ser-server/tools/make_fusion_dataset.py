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


import glob


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SER example')
    parser.add_argument('--rootdir', default='/Data/haradatm/DATA/IEMOCAP/IEMOCAP_full_release', type=str, help='path to datasets')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    for speaker in sorted(os.listdir(args.rootdir)):
        if speaker[0] == 'S':
            wav_dir = os.path.join(args.rootdir, speaker, 'sentences/wav')
            txt_dir = os.path.join(args.rootdir, speaker, 'dialog/transcriptions')
            emo_dir = os.path.join(args.rootdir, speaker, 'dialog/EmoEvaluation')

            for sess in sorted(os.listdir(wav_dir)):
                if sess[7] == 'i':
                    emo_file = emo_dir + '/' + sess + '.txt'
                    txt_file = txt_dir + '/' + sess + '.txt'

                    emo_map = {}
                    for i, line in enumerate(open(emo_file, 'r')):
                        line = line.strip()
                        if line == '':
                            continue
                        if line.startswith('['):
                            cols = line.split()
                            emo_map[cols[3]] = cols[4]

                    for i, line in enumerate(open(txt_file, 'r')):
                        line = line.strip()
                        if line == '':
                            continue
                        if not line.startswith('Ses'):
                            continue

                        cols = line.split()
                        sent = cols[0]
                        wave = speaker + '/sentences/wav/' + sess + '/' + sent + ".wav"
                        text = ' '.join(cols[2:])
                        try:
                            emot = emo_map[sent]
                        except KeyError as e:
                            logger.exception(e)

                        print("{}\t{}\t{}\t{}".format(sent, wave, text, emot))
                        sys.stdout.flush()

if __name__ == '__main__':
    main()
    logger.info('time spent: {:.6f} sec\n'.format(time.time() - start_time))
