# EXAMPLES

This file provides intructions on how to train parsing models from scratch and evaluate them.
Some information has been given in [`README`](README.md).
Here we describe in detail the commands and other settings.

## Dependency Parsing

Below are examples of training `biaffine`  and `crf2o` dependency parsers on PTB.

```sh
# biaffine
$ python -u -m supar.cmds.biaffine_dep train -b -d 0 -c biaffine-dep-en -p model -f char  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \ 
    --embed glove.6B.100d.txt  \ 
    --unk 
# crf2o
$ python -u -m supar.cmds.crf2o_dep train -b -d 0 -c crf2o-dep-en -p model -f char  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \ 
    --embed glove.6B.100d.txt  \ 
    --unk unk  \
    --mbr  \
    --proj
```
The option `-c` controls where to load predefined configs, you can either specify a local file path or the same short name as a pretrained model.
For CRF models, you need to specify `--proj` to remove non-projective trees. 
Specifying `--mbr` to perform MBR decoding often leads to consistent improvement.

The English model finetuned on [`robert-large`](https://huggingface.co/roberta-large) achieves nearly state-of-the-art performance.
Here we provide some recommended hyper-parameters (not the best, but good enough).
You are allowed to set values of registered/unregistered parameters on the command line to suppress default configs.
```sh
$ python -u -m supar.cmds.biaffine_dep train -b -d 5 -c biaffine-dep-roberta-en -p model  \
    --train ptb/train.conllx  \
    --dev ptb/dev.conllx  \
    --test ptb/test.conllx  \ 
    --encoder=bert  \
    --bert=roberta-large  \
    --lr=5e-5  \
    --lr-rate=20  \
    --batch-size=5000  \
    --epochs=10  \
    --update-steps=4
```

To evaluate
```sh
# biaffine
python -u -m supar.cmds.biaffine_dep evaluate -d 0 --data ptb/test.conllx --tree  --proj
# crf2o
python -u -m supar.cmds.crf2o_dep evaluate -d 0  --data ptb/test.conllx --mbr --tree --proj
```
`--tree` and `--proj` ensures to output well-formed and projective trees respectively. 

The commands for training and evaluating Chinese models are similar, except that you need to specify `--punct` to include punctuation.

## Constituency Parsing

```sh
```

## Semantic Dependency Parsing

```sh
```