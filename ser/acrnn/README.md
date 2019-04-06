# SER (Speech Emotion Recognition) example using CNN and Attention

### Description

This example code is a speech emotion recognition using CNN and Attention mechanism based on the following two papers.
- ["3-D Convolutional Recurrent Neural Networks with Attention Model for Speech Emotion Recognition" by Mingyi Chen, Xuanji He, Jing Yang, and Han Zhang, IEEE Signal Processing Letters, vol. 25, issue 10, pp. 1440-1444](https://github.com/xuanjihe/speech-emotion-recognition/blob/master/3-D.pdf)

### Dependencies
- python 3.6
- chainer 3.4

In addition, please add the project folder to PYTHONPATH and `conca install` the following packages:
- `matplotlib`
- `wave, python_speech_features`
- `pickle, bz2`

### Usage ###

***Data***

  - Downlod [IEMOCAP Datasets](https://sail.usc.edu/iemocap/release_form.php) and put them in `datasets/IEMOCAP_full_release`.

***Run and Evaluate***

- Training

```
python train_IEMOCAP-2X.py --gpu 0 --batchsize 40 --dropout 0.4 --epoch 300 --train datasets/iemocap/05-train.txt --valid datasets/iemocap/05-valid.txt --test datasets/iemocap/05-test.txt --out 05-fbank-20-300_b020_e300_d010_adam --optim adam --type fbank --datasets_rootdir datasets/IEMOCAP_full_release 2>&1 | tee 05-fbank-2X-300_b020_e300_d040_adam.log
```

- Evaluating (for early_stopped, final model)

```
python test_IEMOCAP-2X.py --gpu 0 --batchsize 1 --model iemocap-fbank-2X-300_b020_e300_d040_adam/model/final.model --label features/iemocap-fbank-20/labels.pkl --mean features/iemocap-fbank-20/mean-std.pkl --test datasets/iemocap/05-test.txt --out iemocap-fbank-20-300_b020_e300_d040_adam-final-test --type fbank --datasets_rootdir datasets/IEMOCAP_full_release 2>&1 | tee iemocap-fbank-20-300_b020_e300_d040_adam-final-test.log
python test_IEMOCAP-2X.py --gpu 0 --batchsize 1 --model iemocap-fbank-2X-300_b020_e300_d040_adam/model/early_stopped.model --label features/iemocap-fbank-20/labels.pkl --mean features/iemocap-fbank-20/mean-std.pkl --test datasets/iemocap/05-test.txt --out iemocap-fbank-20-300_b020_e300_d040_adam-early_stopped-test --type fbank --datasets_rootdir datasets/IEMOCAP_full_release 2>&1 | tee iemocap-fbank-20-300_b020_e300_d040_adam-early_stopped-test.log
```
