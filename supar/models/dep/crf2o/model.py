# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from supar.config import Config
from supar.models.dep.biaffine.model import BiaffineDependencyModel
from supar.models.dep.biaffine.transform import CoNLL
from supar.modules import Biaffine, MLP, Triaffine
from supar.structs import Dependency2oCRF, MatrixTree
from supar.utils.common import MIN


class CRF2oDependencyModel(BiaffineDependencyModel):
    r"""
    The implementation of second-order CRF Dependency Parser :cite:`zhang-etal-2020-efficient`.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_sib_mlp (int):
            Sibling MLP size. Default: 100.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_rels,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 n_encoder_hidden=800,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_sib_mlp=100,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.arc_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.sib_mlp_s = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.sib_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_sib_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.sib_attn = Triaffine(n_in=n_sib_mlp, scale=scale, bias_x=True, bias_y=True)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor:
                Scores of all possible arcs (``[batch_size, seq_len, seq_len]``),
                dependent-head-sibling triples (``[batch_size, seq_len, seq_len, seq_len]``) and
                all possible labels on each arc (``[batch_size, seq_len, seq_len, n_labels]``).
        """

        x = self.encode(words, feats)
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        sib_s = self.sib_mlp_s(x)
        sib_d = self.sib_mlp_d(x)
        sib_h = self.sib_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        # [batch_size, seq_len, seq_len, seq_len]
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_sib, s_rel

    def loss(self, s_arc, s_sib, s_rel, arcs, sibs, rels, mask, mbr=True, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            sibs (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard siblings.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """

        arc_dist = Dependency2oCRF((s_arc, s_sib), mask.sum(-1))
        arc_loss = -arc_dist.log_prob((arcs, sibs), partial=partial).sum() / mask.sum()
        if mbr:
            s_arc, s_sib = arc_dist.marginals
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, s_arc, s_sib

    def decode(self, s_arc, s_sib, s_rel, mask, tree=False, mbr=True, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i + 1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            if proj:
                arc_preds[bad] = Dependency2oCRF((s_arc[bad], s_sib[bad]), mask[bad].sum(-1)).argmax
            else:
                arc_preds[bad] = MatrixTree(s_arc[bad], mask[bad].sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
