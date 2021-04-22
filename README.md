# SuPar

[![build](https://github.com/yzhangcs/parser/workflows/build/badge.svg)](https://github.com/yzhangcs/parser/actions)
[![docs](https://readthedocs.org/projects/parser/badge/?version=latest)](https://parser.readthedocs.io/en/latest)
[![release](https://img.shields.io/github/v/release/yzhangcs/parser)](https://yzhangcs/parser/releases)
[![downloads](https://pepy.tech/badge/supar)](https://pepy.tech/project/supar)
[![LICENSE](https://img.shields.io/github/license/yzhangcs/parser)](https://github.com/yzhangcs/parser/blob/master/LICENSE)

A Python package that includes many state-of-the-art syntactic/semantic parsers (with pretrained models for more than 19 languages), as well as highly-parallelized implementations of several well-known and effective structured prediction algorithms.

* Dependency Parser
  * Biaffine ([Dozat and Manning, 2017](https://parser.readthedocs.io/en/latest/refs.html#dozat-2017-biaffine))
  * CRF/MatrixTree ([Koo et al., 2007](https://parser.readthedocs.io/en/latest/refs.html#koo-2007-structured); [Ma and Hovy, 2017](https://parser.readthedocs.io/en/latest/refs.html#ma-2017-neural))
  * CRF2o ([Zhang et al., 2020a](https://parser.readthedocs.io/en/latest/refs.html#zhang-2020-efficient))
* Constituency Parser
  * CRF ([Zhang et al., 2020b](https://parser.readthedocs.io/en/latest/refs.html#zhang-2020-fast))
* Semantic Dependency Parser
  * Biaffine ([Dozat and Manning, 2018](https://parser.readthedocs.io/en/latest/refs.html#wang-2019-second))
  * MFVI/LBP ([Wang et al, 2019](https://parser.readthedocs.io/en/latest/refs.html#wang-2019-second))

## Installation

`SuPar` can be installed via pip:
```sh
$ pip install -U supar
```
Or installing from source is also permitted:
```sh
$ git clone https://github.com/yzhangcs/parser && cd parser
$ python setup.py install
```

As a prerequisite, the following requirements should be satisfied:
* `python`: >= 3.7
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.7
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0

## Performance

`SuPar` provides pretrained models for English, Chinese and 17 other languages.
The tables below list the performance and parsing speed of pretrained models for different tasks.
All results are tested on the machine with Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz and Nvidia GeForce GTX 1080 Ti GPU.

### Dependency Darsing

English and Chinese dependency parsing models are trained on PTB and CTB7 respectively.
During evaluation, punctuation is ignored in all metrics for PTB.

| Name                      |  UAS  |  LAS  | Sents/s |
| ------------------------- | :---: | :---: | ------: |
| `biaffine-dep-en`         | 96.01 | 94.41 | 1831.91 |
| `crf-dep-en`              | 96.02 | 94.42 |  762.84 |
| `crf2o-dep-en`            | 96.07 | 94.51 |  531.59 |
| `biaffine-dep-roberta-en` | 97.33 | 95.86 |  271.80 |
| `biaffine-dep-zh`         | 88.64 | 85.47 | 1180.57 |
| `crf-dep-zh`              | 88.75 | 85.65 |  383.97 |
| `crf2o-dep-zh`            | 89.22 | 86.15 |  237.40 |
| `biaffine-dep-electra-zh` | 92.20 | 89.10 |  160.56 |

The multilingual dependency parsing model, named as `biaffine-dep-xlmr`, is trained on the merged 12 selected treebanks from Universal Dependencies (UD) v2.3 dataset by finetuning `xlm-roberta-large` from [Hugingface Transformers](https://github.com/huggingface/transformers).
The following table lists the results of each treebank.
We use [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) to represent these languages.

| Language |  UAS  |  LAS  | Sents/s |
| -------- | :---: | :---: | ------: |
| `bg`     | 96.95 | 94.24 |  343.96 |
| `ca`     | 95.57 | 94.20 |  184.88 |
| `cs`     | 95.79 | 93.83 |  245.68 |
| `de`     | 89.74 | 85.59 |  283.53 |
| `en`     | 93.37 | 91.27 |  269.16 |
| `es`     | 94.78 | 93.29 |  192.00 |
| `fr`     | 94.56 | 91.90 |  219.35 |
| `it`     | 96.29 | 94.47 |  254.82 |
| `nl`     | 96.04 | 93.76 |  268.57 |
| `no`     | 95.64 | 94.45 |  318.00 |
| `ro`     | 94.59 | 89.79 |  216.45 |
| `ru`     | 96.37 | 95.24 |  243.56 |

### Constituency Darsing

| Name                 |   P   |   R   | F<sub>1 | Sents/s |
| -------------------- | :---: | :---: | :-----: | ------: |
| `crf-con-en`         | 94.16 | 93.98 |  94.07  |  841.88 |
| `crf-con-roberta-en` | 96.44 | 96.05 |  96.25  |  233.34 |
| `crf-con-zh`         | 88.82 | 88.42 |  88.62  |  590.05 |
| `crf-con-electra-zh` | 92.15 | 91.56 |  91.85  |  140.45 |

Following [Benepar](https://github.com/nikitakit/self-attentive-parser), the multilingual model `crf-con-xlmr` is trained on SPMRL dataset by finetuning `xlm-roberta-large`.
For simplicity, we directly merge the train/dev/test treebanks of all languages in SPMRL into big ones.
The results of each treebank are as follows. 

| Language |   P   |   R   | F<sub>1 | Sents/s |
| -------- | :---: | :---: | :-----: | ------: |
| `eu`     | 93.40 | 94.19 |  93.79  |  266.96 |
| `fr`     | 88.77 | 88.84 |  88.81  |  149.34 |
| `de`     | 93.68 | 92.18 |  92.92  |  200.31 |
| `he`     | 94.65 | 95.20 |  94.93  |  172.50 |
| `hu`     | 96.70 | 96.81 |  96.76  |  186.58 |
| `ko`     | 91.75 | 92.46 |  92.11  |  234.86 |
| `pl`     | 97.33 | 97.27 |  97.30  |  310.86 |
| `sv`     | 92.51 | 92.50 |  92.50  |  235.49 |

### Semantic Dependency Darsing

| Name                      |   P   |   R   | F<sub>1 | Sents/s |
| ------------------------- | :---: | :---: | :-----: | ------: |
| `biaffine-sdp-en`         | 94.35 | 93.12 |  93.73  | 1067.06 |
| `vi-sdp-en`               | 94.36 | 93.52 |  93.94  |  821.73 |
| `biaffine-sdp-roberta-en` | 94.97 | 95.22 |  95.09  |  266.44 |

## Usage

`SuPar` allows you to download the pretrained model and parse sentences with a few lines of code:
```py
>>> from supar import Parser
>>> parser = Parser.load('biaffine-dep-en')
>>> dataset = parser.predict('She enjoys playing tennis.', prob=True, verbose=False)
100%|####################################| 1/1 00:00<00:00, 85.15it/s
```
The call to `parser.predict` will return an instance of `supar.utils.Dataset` containing the predicted results.
You can either access each sentence held in `dataset` or an individual field of all results.
```py
>>> print(dataset.sentences[0])
1       She     _       _       _       _       2       nsubj   _       _
2       enjoys  _       _       _       _       0       root    _       _
3       playing _       _       _       _       2       xcomp   _       _
4       tennis  _       _       _       _       3       dobj    _       _
5       .       _       _       _       _       2       punct   _       _

>>> print(f"arcs:  {dataset.arcs[0]}\n"
          f"rels:  {dataset.rels[0]}\n"
          f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")
arcs:  [2, 0, 2, 3, 2]
rels:  ['nsubj', 'root', 'xcomp', 'dobj', 'punct']
probs: tensor([1.0000, 0.9999, 0.9642, 0.9686, 0.9996])
```
Probabilities can be returned along with the results if `prob=True`.
For CRF parsers, marginals are available if `mbr=True`, i.e., using MBR decoding.

If you'd like to parse un-tokenized raw texts, you can call `nltk.word_tokenize` to do the tokenization first:
```py
>>> import nltk
>>> text = nltk.word_tokenize('She enjoys playing tennis.')
>>> print(parser.predict([text], verbose=False).sentences[0])
100%|####################################| 1/1 00:00<00:00, 74.20it/s
1       She     _       _       _       _       2       nsubj   _       _
2       enjoys  _       _       _       _       0       root    _       _
3       playing _       _       _       _       2       xcomp   _       _
4       tennis  _       _       _       _       3       dobj    _       _
5       .       _       _       _       _       2       punct   _       _

```

If there are a plenty of sentences to parse, `SuPar` also supports for loading them from file, and save to the `pred` file if specified.
```py
>>> dataset = parser.predict('data/ptb/test.conllx', pred='pred.conllx')
2020-07-25 18:13:50 INFO Loading the data
2020-07-25 18:13:52 INFO
Dataset(n_sentences=2416, n_batches=13, n_buckets=8)
2020-07-25 18:13:52 INFO Making predictions on the dataset
100%|####################################| 13/13 00:01<00:00, 10.58it/s
2020-07-25 18:13:53 INFO Saving predicted results to pred.conllx
2020-07-25 18:13:54 INFO 0:00:01.335261s elapsed, 1809.38 Sents/s
```

Please make sure the file is in CoNLL-X format. If some fields are missing, you can use underscores as placeholders.
An interface is provided for the transformation from text to CoNLL-X format string.
```py
>>> from supar.utils import CoNLL
>>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
1       She     _       _       _       _       _       _       _       _
2       enjoys  _       _       _       _       _       _       _       _
3       playing _       _       _       _       _       _       _       _
4       tennis  _       _       _       _       _       _       _       _
5       .       _       _       _       _       _       _       _       _

```

For Universial Dependencies (UD), the CoNLL-U file is also allowed, while comment lines in the file can be reserved before prediction and recovered during post-processing.
```py
>>> import os
>>> import tempfile
>>> text = '''# text = But I found the location wonderful and the neighbors very kind.
1\tBut\t_\t_\t_\t_\t_\t_\t_\t_
2\tI\t_\t_\t_\t_\t_\t_\t_\t_
3\tfound\t_\t_\t_\t_\t_\t_\t_\t_
4\tthe\t_\t_\t_\t_\t_\t_\t_\t_
5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_
6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_
7\tand\t_\t_\t_\t_\t_\t_\t_\t_
7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_
8\tthe\t_\t_\t_\t_\t_\t_\t_\t_
9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_
10\tvery\t_\t_\t_\t_\t_\t_\t_\t_
11\tkind\t_\t_\t_\t_\t_\t_\t_\t_
12\t.\t_\t_\t_\t_\t_\t_\t_\t_

'''
>>> path = os.path.join(tempfile.mkdtemp(), 'data.conllx')
>>> with open(path, 'w') as f:
...     f.write(text)
...
>>> print(parser.predict(path, verbose=False).sentences[0])
100%|####################################| 1/1 00:00<00:00, 68.60it/s
# text = But I found the location wonderful and the neighbors very kind.
1       But     _       _       _       _       3       cc      _       _
2       I       _       _       _       _       3       nsubj   _       _
3       found   _       _       _       _       0       root    _       _
4       the     _       _       _       _       5       det     _       _
5       location        _       _       _       _       6       nsubj   _       _
6       wonderful       _       _       _       _       3       xcomp   _       _
7       and     _       _       _       _       6       cc      _       _
7.1     found   _       _       _       _       _       _       _       _
8       the     _       _       _       _       9       det     _       _
9       neighbors       _       _       _       _       11      dep     _       _
10      very    _       _       _       _       11      advmod  _       _
11      kind    _       _       _       _       6       conj    _       _
12      .       _       _       _       _       3       punct   _       _

```

Constituency trees can be parsed in a similar manner.
The returned `dataset` holds all predicted trees represented using `nltk.Tree` objects.
```py
>>> parser = Parser.load('crf-con-en')
>>> dataset = parser.predict([['She', 'enjoys', 'playing', 'tennis', '.']], verbose=False)
100%|####################################| 1/1 00:00<00:00, 75.86it/s
>>> print(f"trees:\n{dataset.trees[0]}")
trees:
(TOP
  (S
    (NP (_ She))
    (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
    (_ .)))
>>> dataset = parser.predict('data/ptb/test.pid', pred='pred.pid')
2020-07-25 18:21:28 INFO Loading the data
2020-07-25 18:21:33 INFO
Dataset(n_sentences=2416, n_batches=13, n_buckets=8)
2020-07-25 18:21:33 INFO Making predictions on the dataset
100%|####################################| 13/13 00:02<00:00,  5.30it/s
2020-07-25 18:21:36 INFO Saving predicted results to pred.pid
2020-07-25 18:21:36 INFO 0:00:02.455740s elapsed, 983.82 Sents/s
```
### Training

To train a model from scratch, it is preferred to use the command-line option, which is more flexible and customizable.
Here are some training examples:
```sh
# Biaffine Dependency Parser
# some common and default arguments are stored in config.ini
$ python -m supar.cmds.biaffine_dependency train -b -d 0  \
    -c config.ini  \
    -p exp/ptb.biaffine.dependency.char/model  \
    -f char
# to use BERT, `-f` and `--bert` (default to bert-base-cased) should be specified
# if you'd like to use XLNet, you can type `--bert xlnet-base-cased`
$ python -m supar.cmds.biaffine_dependency train -b -d 0  \
    -p exp/ptb.biaffine.dependency.bert/model  \
    -f bert  \
    --bert bert-base-cased

# CRF Dependency Parser
# for CRF dependency parsers, you should use `--proj` to discard all non-projective training instances
# optionally, you can use `--mbr` to perform MBR decoding
$ python -m supar.cmds.crf_dependency train -b -d 0  \
    -p exp/ptb.crf.dependency.char/model  \
    -f char  \
    --mbr  \
    --proj

# CRF Constituency Parser
# the training of CRF constituency parser behaves like dependency parsers
$ python -m supar.cmds.crf_constituency train -b -d 0  \
    -p exp/ptb.crf.constituency.char/model -f char  \
    --mbr
```

For more instructions on training, please type `python -m supar.cmds.<parser> train -h`.

Alternatively, `SuPar` provides some equivalent command entry points registered in `setup.py`:
`biaffine-dependency`, `crfnp-dependency`, `crf-dependency`, `crf2o-dependency` and `crf-constituency`.
```sh
$ biaffine-dependency train -b -d 0 -c config.ini -p exp/ptb.biaffine.dependency.char/model -f char
```

To accommodate large models, distributed training is also supported:
```sh
$ python -m torch.distributed.launch --nproc_per_node=4 --master_port=10000  \
    -m supar.cmds.biaffine_dependency train -b -d 0,1,2,3  \
    -p exp/ptb.biaffine.dependency.char/model  \
    -f char
```
You can consult the PyTorch [documentation](https://pytorch.org/docs/stable/notes/ddp.html) and [tutorials](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

### Evaluation

The evaluation process resembles prediction:
```py
>>> parser = Parser.load('biaffine-dep-en')
>>> loss, metric = parser.evaluate('data/ptb/test.conllx')
2020-07-25 20:59:17 INFO Loading the data
2020-07-25 20:59:19 INFO
Dataset(n_sentences=2416, n_batches=11, n_buckets=8)
2020-07-25 20:59:19 INFO Evaluating the dataset
2020-07-25 20:59:20 INFO loss: 0.2326 - UCM: 61.34% LCM: 50.21% UAS: 96.03% LAS: 94.37%
2020-07-25 20:59:20 INFO 0:00:01.253601s elapsed, 1927.25 Sents/s
```

## Citation

The CRF models for Dependency/Constituency parsing are our recent works published in ACL 2020 and IJCAI 2020 respectively.
If you are interested in them, please cite:
```bib
@inproceedings{zhang-etal-2020-efficient,
  title     = {Efficient Second-Order {T}ree{CRF} for Neural Dependency Parsing},
  author    = {Zhang, Yu and Li, Zhenghua and Zhang Min},
  booktitle = {Proceedings of ACL},
  year      = {2020},
  url       = {https://www.aclweb.org/anthology/2020.acl-main.302},
  pages     = {3295--3305}
}

@inproceedings{zhang-etal-2020-fast,
  title     = {Fast and Accurate Neural {CRF} Constituency Parsing},
  author    = {Zhang, Yu and Zhou, Houquan and Li, Zhenghua},
  booktitle = {Proceedings of IJCAI},
  year      = {2020},
  doi       = {10.24963/ijcai.2020/560},
  url       = {https://doi.org/10.24963/ijcai.2020/560},
  pages     = {4046--4053}
}
```
