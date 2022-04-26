# Language identification experiments using the VoxLingua107 dataset

This folder contains scripts for running language identification experiments with the VoxLingua107 dataset.
These experiments were highly inspired by the language identification experiments on the CommonLanguage dataset,
as well as speaker recognition experiments with the VoxCeleb dataset. However, due to the large size of
the VoxLingua107 dataset, it has also significant differences, most prominently in using a WebDataset
based data loading pipeline. Also, the models use more feed-forward layers after the utterance embedding layer,
and cross-entropy loss instead of additive angular margin loss, as this was found to improve the quality of
the embeddings when used as features in a supervised language identification task.

The VoxLingua107 dataset contains over 2.5 million utterance-like audio files. Training a model on them
as in the Voxceleb recipe would cause heavy disk loads. Therefore we opted to using WebDataset based training:
before training, audio files are shuffled and distributed into over 500 so-called shards (tar files). During training,
the tar files are opened in random order, the audio files in the shards are shuffled again on-the-fly using a moderately large buffer
and fed to the training process. This reduces the disk load during training by large margin. This is all
handled by the WebDataset library.

## Downloading the data

You have two options how to download and prepare the VoxLingua107 dataset for training the model:

  - Download the VoxLingua107 language-specific zips from http://bark.phon.ioc.ee/voxlingua107/ and convert them
    to WebDataset format. This is the most flexible option, as it allows selecting a subset of VoxLingua107 languages,
    or adding new languages. It will require around 2.2 TB disk space.

  - Download the pre-compiled WebDataset shards from http://bark.phon.ioc.ee/voxlingua107/. It will require around 1.4T of disk space.


### 1st option: download the VoxLingua107 zips and create the Webdataset shards

Download the zips:

```
# Select a place with around 1 TB of free space
cd /data/
mkdir voxlingua107
cd voxlingua107
wget http://bark.phon.ioc.ee/voxlingua107/zip_urls.txt
cat zip_urls.txt | xargs  wget --continue
wget bark.phon.ioc.ee/voxlingua107/dev.zip

```

Create WebDataset shards:

```
python create_wds_shards.py /data/voxlingua107/train/ /data/voxlingua107_shards/train
python create_wds_shards.py /data/voxlingua107/dev/ /data/voxlingua107_shards/dev
```

### 2nd option: download the pre-compiled WebDataset shards

Download the shards:

```
# Select a place with around 1 TB of free space
cd /data/
mkdir voxlingua107_shards
cd voxlingua107_shards
wget  -r -nH --cut-dirs=4 --no-parent --reject="index.html*" http://bark.phon.ioc.ee/lw/korpused/voxlingua107/shards/
```


## Training

```
python train.py hparams/train_ecapa.yaml
```

```
python train_wav2vec.py hparams/train_wav2vec.yaml
```

Training is run for 40 epochs. One epoch takes one hour and 40 minutes on a NVidia A100 GPU.


# Performance
| Release | hyperparams file | Dev error rate | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| :-----------:|
| 21-08-24 | train_ecapa.yaml | 6.7 |https://drive.google.com/drive/folders/151QTW9oHVElLIkuzXjkuHpOCLNZF0Ufd?usp=sharing | 1xA100 40GB |
| 22-04-14 | train_wav2vec.yaml | 5.6 |https://huggingface.co/TalTechNLP/voxlingua107-xls-r-300m-wav2vec| 2xA100 40GB |



# Inference
The pre-trained model + easy inference is available on HuggingFace:
- https://huggingface.co/TalTechNLP/voxlingua107-xls-r-300m-wav2vec
- https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa

You can run inference with only few lines of code:

## TalTechNLP/voxlingua107-xls-r-300m-wav2vec:

_This model uses custom inference classifier and hparams file included in HuggingFace._

```python
import torchaudio
from speechbrain.pretrained.interfaces import foreign_class
language_id = foreign_class(source="TalTechNLP/voxlingua107-xls-r-300m-wav2vec", pymodule_file="encoder_wav2vec_classifier.py", classname="EncoderWav2vecClassifier", hparams_file='inference_wav2vec.yaml', savedir="tmp")
# Download Thai language sample from Omniglot and convert to suitable form
wav_file = "https://omniglot.com/soundfiles/udhr/udhr_th.mp3"
out_prob, score, index, text_lab = language_id.classify_file(wav_file)
print("probability:", out_prob)
print("label:", text_lab)
print("score:", score)
print("index:", index)
    probability: tensor([[[-2.2849e+01, -2.4349e+01, -2.3686e+01, -2.3632e+01, -2.0218e+01,
            -2.7241e+01, -2.6715e+01, -2.2301e+01, -2.6076e+01, -2.1716e+01,
            -1.9923e+01, -2.7303e+01, -2.1211e+01, -2.2998e+01, -2.4436e+01,
            -2.6437e+01, -2.2686e+01, -2.4244e+01, -2.0416e+01, -2.8329e+01,
            -1.7788e+01, -2.4829e+01, -2.4186e+01, -2.7036e+01, -2.5993e+01,
            -1.9677e+01, -2.2746e+01, -2.9192e+01, -2.4941e+01, -2.7135e+01,
            -2.6653e+01, -2.2791e+01, -2.4599e+01, -2.1066e+01, -2.4855e+01,
            -2.1874e+01, -2.2914e+01, -2.4174e+01, -2.0902e+01, -2.3197e+01,
            -2.6108e+01, -2.3941e+01, -2.3103e+01, -2.2363e+01, -2.8969e+01,
            -2.5302e+01, -2.4862e+01, -2.2392e+01, -2.4042e+01, -2.1221e+01,
            -2.3656e+01, -2.1286e+01, -1.9209e+01, -2.3254e+01, -2.8291e+01,
            -5.9105e+00, -2.4525e+01, -2.4937e+01, -2.8349e+01, -2.4420e+01,
            -2.7439e+01, -2.6329e+01, -2.3317e+01, -2.3842e+01, -2.2114e+01,
            -2.3637e+01, -1.7217e+01, -1.8342e+01, -2.4332e+01, -2.6090e+01,
            -2.5452e+01, -2.3854e+01, -2.6082e+01, -2.4992e+01, -2.0618e+01,
            -2.9351e+01, -2.4153e+01, -2.3156e+01, -2.6893e+01, -2.5314e+01,
            -2.8374e+01, -2.4009e+01, -2.3604e+01, -2.4063e+01, -2.3538e+01,
            -2.4953e+01, -2.5607e+01, -2.3960e+01, -2.6471e+01, -2.3348e+01,
            -2.1681e+01, -2.7610e+01, -2.5023e+01, -2.3585e+01, -2.7146e-03,
            -2.0338e+01, -1.8737e+01, -2.5158e+01, -2.7491e+01, -2.3623e+01,
            -2.5718e+01, -2.3465e+01, -1.8305e+01, -2.1064e+01, -2.9880e+01,
            -2.2809e+01, -1.9856e+01]]])
    # The identified language ISO code is given in score[0][0]
    label: [['th']]
    score: tensor([[-0.0027]])
    index: tensor([[94]])
# The scores in the out_prob tensor can be interpreted as log-likelihoods that
# the given utterance belongs to the given language (i.e., the larger the better)
# The linear-scale likelihood can be retrieved using the following:
print(score.exp())
  tensor([0.9973])
  
# Alternatively, use the utterance embedding extractor:
signal, fs = torchaudio.load(wav_file)
embeddings =  language_id.encode_batch(signal)
print(embeddings.shape)
    torch.Size([2, 1, 2048])
```

## speechbrain/lang-id-voxlingua107-ecapa:
```python
import torchaudio
from speechbrain.pretrained import EncoderClassifier
language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
# Download Thai language sample from Omniglot and cvert to suitable form
signal = language_id.load_audio("https://omniglot.com/soundfiles/udhr/udhr_th.mp3")
prediction =  language_id.classify_batch(signal)
print(prediction)
  (tensor([[-2.8646e+01, -3.0346e+01, -2.0748e+01, -2.9562e+01, -2.2187e+01,
         -3.2668e+01, -3.6677e+01, -3.3573e+01, -3.2545e+01, -2.4365e+01,
         -2.4688e+01, -3.1171e+01, -2.7743e+01, -2.9918e+01, -2.4770e+01,
         -3.2250e+01, -2.4727e+01, -2.6087e+01, -2.1870e+01, -3.2821e+01,
         -2.2128e+01, -2.2822e+01, -3.0888e+01, -3.3564e+01, -2.9906e+01,
         -2.2392e+01, -2.5573e+01, -2.6443e+01, -3.2429e+01, -3.2652e+01,
         -3.0030e+01, -2.4607e+01, -2.2967e+01, -2.4396e+01, -2.8578e+01,
         -2.5153e+01, -2.8475e+01, -2.6409e+01, -2.5230e+01, -2.7957e+01,
         -2.6298e+01, -2.3609e+01, -2.5863e+01, -2.8225e+01, -2.7225e+01,
         -3.0486e+01, -2.1185e+01, -2.7938e+01, -3.3155e+01, -1.9076e+01,
         -2.9181e+01, -2.2160e+01, -1.8352e+01, -2.5866e+01, -3.3636e+01,
         -4.2016e+00, -3.1581e+01, -3.1894e+01, -2.7834e+01, -2.5429e+01,
         -3.2235e+01, -3.2280e+01, -2.8786e+01, -2.3366e+01, -2.6047e+01,
         -2.2075e+01, -2.3770e+01, -2.2518e+01, -2.8101e+01, -2.5745e+01,
         -2.6441e+01, -2.9822e+01, -2.7109e+01, -3.0225e+01, -2.4566e+01,
         -2.9268e+01, -2.7651e+01, -3.4221e+01, -2.9026e+01, -2.6009e+01,
         -3.1968e+01, -3.1747e+01, -2.8156e+01, -2.9025e+01, -2.7756e+01,
         -2.8052e+01, -2.9341e+01, -2.8806e+01, -2.1636e+01, -2.3992e+01,
         -2.3794e+01, -3.3743e+01, -2.8332e+01, -2.7465e+01, -1.5085e-02,
         -2.9094e+01, -2.1444e+01, -2.9780e+01, -3.6046e+01, -3.7401e+01,
         -3.0888e+01, -3.3172e+01, -1.8931e+01, -2.2679e+01, -3.0225e+01,
         -2.4995e+01, -2.1028e+01]]), tensor([-0.0151]), tensor([94]), ['th'])
# The scores in the prediction[0] tensor can be interpreted as log-likelihoods that
# the given utterance belongs to the given language (i.e., the larger the better)
# The linear-scale likelihood can be retrieved using the following:
print(prediction[1].exp())
  tensor([0.9850])
# The identified language ISO code is given in prediction[3]
print(prediction[3])
  ['th']

# Alternatively, use the utterance embedding extractor:
emb =  language_id.encode_batch(signal)
print(emb.shape)
  torch.Size([1, 1, 256])
```


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

# **Citing VoxLingua107**
You can also cite the VoxLingua107 dataset paper if you use this model in research.

```bibtex
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}
```
