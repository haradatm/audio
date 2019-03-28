# Real-time SER (Speech Emotion Recognition) example using OpenSMILE features

### Description

This example code is a speech emotion recognition using OpenSMILE and Bert Embeddings features.
See also 

### Dependencies
- python 3.6
- chainer 3.4

In addition, please add the project folder to PYTHONPATH and `conca install` the following packages:
- `matplotlib`
- `wave, python_speech_features`
- `pickle, bz2`
- `tornado`

### ToDo ###
- Support opus codec 
- Visualize emotions

### Usage ###

***Data***

  - Downlod [IEMOCAP Datasets](https://sail.usc.edu/iemocap/release_form.php) and put them in `datasets/IEMOCAP_full_release`.

```
python tools/make_fusion_dataset.py --rootdir IEMOCAP_full_release > datasets/iemocap/all.txt
wc -l datasets/iemocap/all.txt
    4829 datasets/iemocap/all.txt

cat datasets/iemocap/all.txt | grep -v "^Ses05" | grep -E "neu$|ang$|sad$|hap$" > datasets/iemocap/05-train.txt
cat datasets/iemocap/all.txt | grep    "^Ses05" | grep -E "neu$|ang$|sad$|hap$" > datasets/iemocap/05-test.txt
wc -l datasets/iemocap/05-{train,test}.txt
    1732 datasets/iemocap/05-train.txt
     570 datasets/iemocap/05-test.txt
    2302 total

python tools/make_feature_smile_dialog.py --input datasets/iemocap/05-train.txt > datasets/iemocap/smile/train-dialog.txt
python tools/make_feature_smile_dialog.py --input datasets/iemocap/05-test.txt  > datasets/iemocap/smile/test-dialog.txt
wc -l datasets/iemocap/smile/*-dialog.txt
      64 datasets/iemocap/smile/train-dialog.txt
      16 datasets/iemocap/smile/test-dialog.txt
      80 total
```

***Train and Evaluate***

```
python train_dialog-lstm.py --use_classweight --gpu 0 --batchsize 8 --epoch 200 --layer 1 --unit 200 --dropout 0.25 \
--train datasets/iemocap/smile/train-dialog.txt \
--eval  datasets/iemocap/smile/test-dialog.txt \
--use_classweight \
--out 05-smile-lstm-0-l1-u300_b008_e200_d025_adam 2>&1 \
| tee 05-smile-lstm-0-l1-u300_b008_e200_d025_adam.log
```

### Realtime Emotion detection ###

- Place the trained model file in `models`.

- Run Server

```
cd server
python server.py --gpu 0 --layer 2 --unit 200 --model models/early_stopped-loss.model
```

- Open http://localhost:8888/ in your Chrome browser and press F12 to access the developer tool 

- Click on `Start` button at the top of windows

- Input your voice with a microphone to estimate your emotions

***Output***

- Output to developer tool console.

```
Vad | sampleRate: 44100 | hertzPerBin: 86.1328125 | iterationFrequency: 86.1328125 | iterationPeriod: 0.011609977324263039
voice start
voice stop
 :
```

- Output to server console log.

```
{
  "gpu": 0,
  "layer": 2,
  "unit": 200,
  "model": "models/early_stopped-uar.model",
}
# class: 4, labels: {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3}
2019-03-27 09:18:28,640 - open - INFO - open
["{'neu': 0.535590}", "{'neu': 0.458634}"]
["{'neu': 0.535590}", "{'neu': 0.458634}", "{'hap': 0.433621}"]
["{'neu': 0.535590}", "{'neu': 0.458634}", "{'hap': 0.433621}", "{'hap': 0.450143}"]
["{'neu': 0.535590}", "{'neu': 0.458634}", "{'hap': 0.433621}", "{'hap': 0.450143}", "{'hap': 0.419426}"]
["{'neu': 0.458634}", "{'hap': 0.433621}", "{'hap': 0.450143}", "{'hap': 0.419426}", "{'hap': 0.433621}"]
 :
2019-03-27 09:19:22,990 - on_close - INFO - on_close
```
