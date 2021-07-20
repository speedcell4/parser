# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.structs.distribution import StructuredDistribution
from supar.structs.semiring import LogSemiring
from supar.utils.alg import mst
from supar.utils.fn import stripe
from torch.distributions.utils import lazy_property


class MatrixTree(StructuredDistribution):
    r"""
    MatrixTree for calculating partitions and marginals of directed spanning trees (a.k.a. non-projective trees)
    in :math:`O(n^3)` by an adaptation of Kirchhoff's MatrixTree Theorem :cite:`koo-etal-2007-structured`.
    """

    def __init__(self, scores, mask=None, multiroot=False, partial=False):
        super().__init__(scores, mask, multiroot=multiroot, partial=partial)

        self.multiroot = multiroot
        self.partial = partial

        self.scores = scores
        self.mask = mask if mask is not None else scores.new_ones(scores.shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores.new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @lazy_property
    def argmax(self):
        with torch.no_grad():
            return mst(self.scores, self.mask, self.multiroot)

    def kmax(self, k):
        raise NotImplementedError

    @lazy_property
    def entropy(self):
        raise NotImplementedError

    def cross_entropy(self, other):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def score(self, value):
        arcs = value
        if self.partial:
            mask, lens = self.mask, self.lens
            mask = mask.index_fill(1, self.lens.new_tensor(0), 1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            arcs = arcs.index_fill(1, lens.new_tensor(0), -1).unsqueeze(-1)
            arcs = arcs.eq(lens.new_tensor(range(mask.shape[1]))) | arcs.lt(0)
            scores = LogSemiring.zero_mask(self.scores, ~(arcs & mask))
            return self.__class__(scores, self.mask, **self.kwargs).log_partition
        return LogSemiring.prod(LogSemiring.one_mask(self.scores.gather(-1, arcs.unsqueeze(-1)).squeeze(-1), ~self.mask), -1)

    @torch.enable_grad()
    def forward(self, semiring):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            value (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs. Default: ``None``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.
        """

        s_arc = self.scores
        mask, lens = self.mask, self.lens
        batch_size, seq_len, _ = s_arc.shape
        mask = mask.index_fill(1, lens.new_tensor(0), 1)
        s_arc = semiring.zero_mask(s_arc, ~(mask.unsqueeze(-1) & mask.unsqueeze(-2)))

        # A(i, j) = exp(s(i, j))
        # double precision to prevent overflows
        A = torch.exp(s_arc).double()
        # Weighted degree matrix
        # D(i, j) = sum_j(A(i, j)), if h == m
        #           0,              otherwise
        D = torch.zeros_like(A)
        D.diagonal(0, 1, 2).copy_(A.sum(-1))
        # Laplacian matrix
        # L(i, j) = D(i, j) - A(i, j)
        L = nn.init.eye_(torch.empty_like(A[0])).repeat(batch_size, 1, 1).masked_scatter_(mask.unsqueeze(-1), (D - A)[mask])
        # Z = L^(0, 0), the minor of L w.r.t row 0 and column 0
        return L[:, 1:, 1:].slogdet()[1].float()


class CRFDependency(StructuredDistribution):
    r"""
    First-order TreeCRF for calculating partitions and marginals of projective dependency trees
    in :math:`O(n^3)` :cite:`zhang-etal-2020-efficient`.
    """

    def __init__(self, scores, mask=None, multiroot=False, partial=False):
        super().__init__(scores, mask, multiroot=multiroot, partial=partial)

        self.multiroot = multiroot
        self.partial = partial

        self.mask = mask if mask is not None else scores.new_ones(scores.shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores.new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @lazy_property
    def argmax(self):
        return self.lens.new_zeros(self.mask.shape).masked_scatter_(self.mask, torch.where(self.backward(self.max.sum()))[2])

    def topk(self, k):
        preds = torch.stack([torch.where(self.backward(i))[2] for i in self.kmax(k).sum(0)], -1)
        return self.lens.new_zeros(*self.mask.shape, k).masked_scatter_(self.mask.unsqueeze(-1), preds)

    def score(self, value):
        arcs = value
        if self.partial:
            mask, lens = self.mask, self.lens
            mask = mask.index_fill(1, self.lens.new_tensor(0), 1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            arcs = arcs.index_fill(1, lens.new_tensor(0), -1).unsqueeze(-1)
            arcs = arcs.eq(lens.new_tensor(range(mask.shape[1]))) | arcs.lt(0)
            scores = LogSemiring.zero_mask(self.scores, ~(arcs & mask))
            return self.__class__(scores, self.mask, **self.kwargs).log_partition
        return LogSemiring.prod(LogSemiring.one_mask(self.scores.gather(-1, arcs.unsqueeze(-1)).squeeze(-1), ~self.mask), -1)

    def forward(self, semiring):
        s_arc = self.scores
        batch_size, seq_len, _ = s_arc.shape[-3:]
        # [..., batch_size, seq_len, seq_len], (h->m)
        s_arc = semiring.convert(s_arc.transpose(-1, -2))
        s_i = semiring.zero_(torch.empty_like(s_arc))
        s_c = semiring.zero_(torch.empty_like(s_arc))
        semiring.one_(s_c.diagonal(0, -2, -1))

        for w in range(1, seq_len):
            n = seq_len - w

            # [..., batch_size, n]
            il = ir = semiring.dot(stripe(s_c, n, w), stripe(s_c, n, w, (w, 1)), -1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w, -2, -1).copy_(semiring.mul(il, s_arc.diagonal(-w, -2, -1)))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w, -2, -1).copy_(semiring.mul(ir, s_arc.diagonal(w, -2, -1)))

            # [..., batch_size, n]
            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            s_c.diagonal(-w, -2, -1).copy_(semiring.dot(stripe(s_c, n, w, (0, 0), 0), stripe(s_i, n, w, (w, 0)), -1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            s_c.diagonal(w, -2, -1).copy_(semiring.dot(stripe(s_i, n, w, (0, 1)), stripe(s_c, n, w, (1, w), 0), -1))
            if not self.multiroot:
                s_c[..., self.lens.ne(w), 0, w] = semiring.zero
        # [..., batch_size, seq_len, seq_len]
        s_c = semiring.unconvert(s_c)
        # [seq_len, batch_size, seq_len, ...]
        s_c = s_c.permute(-2, -3, -1, *range(s_c.dim() - 3))

        return s_c[0][range(batch_size), self.lens]


class CRF2oDependency(StructuredDistribution):
    r"""
    Second-order TreeCRF :cite:`zhang-etal-2020-efficient`.
    """

    def __init__(self, scores, mask=None, multiroot=False, partial=False):
        super().__init__(scores, mask, multiroot=multiroot, partial=partial)

        self.multiroot = multiroot
        self.partial = partial

        self.mask = mask if mask is not None else scores[0].new_ones(scores[0].shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores[0].new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @lazy_property
    def argmax(self):
        return self.lens.new_zeros(self.mask.shape).masked_scatter_(self.mask, torch.where(self.backward(self.max.sum()))[2])

    def topk(self, k):
        preds = torch.stack([torch.where(self.backward(i))[2] for i in self.kmax(k).sum(0)], -1)
        return self.lens.new_zeros(*self.mask.shape, k).masked_scatter_(self.mask.unsqueeze(-1), preds)

    def score(self, value):
        arcs, sibs = value
        if self.partial:
            mask, lens = self.mask, self.lens
            mask = mask.index_fill(1, self.lens.new_tensor(0), 1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            arcs = arcs.index_fill(1, lens.new_tensor(0), -1).unsqueeze(-1)
            arcs = arcs.eq(lens.new_tensor(range(mask.shape[1]))) | arcs.lt(0)
            s_arc, s_sib = LogSemiring.zero_mask(self.scores[0], ~(arcs & mask)), self.scores[1]
            return self.__class__((s_arc, s_sib), self.mask, **self.kwargs).log_partition
        s_arc = self.scores[0].gather(-1, arcs.unsqueeze(-1)).squeeze(-1)
        s_arc = LogSemiring.prod(LogSemiring.one_mask(s_arc, ~self.mask), -1)
        s_sib = self.scores[1].gather(-1, sibs.unsqueeze(-1)).squeeze(-1)
        s_sib = LogSemiring.prod(LogSemiring.one_mask(s_sib, ~sibs.gt(0)), (-1, -2))
        return LogSemiring.mul(s_arc, s_sib)

    @torch.enable_grad()
    def forward(self, semiring):
        s_arc, s_sib = self.scores
        batch_size, seq_len, _ = s_arc.shape[-3:]
        # [..., batch_size, seq_len, seq_len], (h->m)
        s_arc = semiring.convert(s_arc.transpose(-1, -2))
        # [..., batch_size, seq_len, seq_len, seq_len], (h->m->s)
        s_sib = semiring.convert(s_sib.transpose(-2, -3))
        s_i = semiring.zero_(torch.empty_like(s_arc))
        s_s = semiring.zero_(torch.empty_like(s_arc))
        s_c = semiring.zero_(torch.empty_like(s_arc))
        semiring.one_(s_c.diagonal(0, -2, -1))

        for w in range(1, seq_len):
            n = seq_len - w

            # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
            #                  exp(C(j->j) + C(i->j-1)))
            #           + s(j->i)
            # [..., batch_size, n, w]
            il = semiring.times(stripe(s_i, n, w, (w, 1)),
                                stripe(s_s, n, w, (1, 0), 0),
                                stripe(s_sib[..., range(w, n+w), range(n), :], n, w, (0, 1)))
            il[..., -1] = semiring.mul(stripe(s_c, n, 1, (w, w)), stripe(s_c, n, 1, (0, w - 1))).squeeze(-1)
            il = semiring.sum(il, -1)
            s_i.diagonal(-w, -2, -1).copy_(semiring.mul(il, s_arc.diagonal(-w, -2, -1)))
            # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
            #                  exp(C(i->i) + C(j->i+1)))
            #           + s(i->j)
            # [..., batch_size, n, w]
            ir = semiring.times(stripe(s_i, n, w),
                                stripe(s_s, n, w, (0, w), 0),
                                stripe(s_sib[..., range(n), range(w, n+w), :], n, w))
            if not self.multiroot:
                semiring.zero_(ir[..., 0, :])
            ir[..., 0] = semiring.mul(stripe(s_c, n, 1), stripe(s_c, n, 1, (w, 1))).squeeze(-1)
            ir = semiring.sum(ir, -1)
            s_i.diagonal(w, -2, -1).copy_(semiring.mul(ir, s_arc.diagonal(w, -2, -1)))

            # [..., batch_size, n]
            sl = sr = semiring.dot(stripe(s_c, n, w), stripe(s_c, n, w, (w, 1)), -1)
            # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(-w, -2, -1).copy_(sl)
            # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(w, -2, -1).copy_(sr)

            # [..., batch_size, n]
            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            s_c.diagonal(-w, -2, -1).copy_(semiring.dot(stripe(s_c, n, w, (0, 0), 0), stripe(s_i, n, w, (w, 0)), -1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            s_c.diagonal(w, -2, -1).copy_(semiring.dot(stripe(s_i, n, w, (0, 1)), stripe(s_c, n, w, (1, w), 0), -1))
        # [..., batch_size, seq_len, seq_len]
        s_c = semiring.unconvert(s_c)
        # [seq_len, batch_size, seq_len, ...]
        s_c = s_c.permute(-2, -3, -1, *range(s_c.dim() - 3))

        return s_c[0][range(batch_size), self.lens]


class CRFConstituency(StructuredDistribution):
    r"""
    TreeCRF for calculating partitions and marginals of constituency trees :cite:`zhang-etal-2020-fast`.
    """

    def __init__(self, scores, mask=None, labeled=False):
        super().__init__(scores, mask, labeled=labeled)

        self.labeled = labeled

        self.mask = mask if mask is not None else scores.new_ones(scores.shape[:(-1 if labeled else None)]).bool().triu_(1)
        self.lens = self.mask[:, 0].sum(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(labeled={self.labeled})"

    @lazy_property
    def argmax(self):
        return [sorted(torch.nonzero(i).tolist(), key=lambda x:(x[0], -x[1])) for i in self.backward(self.max.sum())]

    def topk(self, k):
        return list(zip(*[[sorted(torch.nonzero(i).tolist(), key=lambda x:(x[0], -x[1])) for i in self.backward(i)]
                          for i in self.kmax(k).sum(0)]))

    def score(self, value):
        return LogSemiring.prod(LogSemiring.prod(LogSemiring.one_mask(self.scores, ~(self.mask & value)), -1), -1)

    @torch.enable_grad()
    def forward(self, semiring):
        scores = semiring.convert(self.scores)
        # [..., batch_size, seq_len, seq_len], (l->r)
        scores = semiring.sum(scores, -1) if self.labeled else scores
        batch_size, seq_len, _ = scores.shape[-3:]
        s = semiring.zero_(torch.empty_like(scores))

        for w in range(1, seq_len):
            n = seq_len - w
            if w == 1:
                s.diagonal(w, -2, -1).copy_(scores.diagonal(w, -2, -1))
                continue
            # [..., batch_size, n]
            s_s = semiring.dot(stripe(s, n, w-1, (0, 1)), stripe(s, n, w-1, (1, w), 0), -1)
            s.diagonal(w, -2, -1).copy_(semiring.mul(s_s, scores.diagonal(w, -2, -1)))
        # [..., batch_size, seq_len, seq_len]
        s = semiring.unconvert(s)
        # [seq_len, batch_size, seq_len, ...]
        s = s.permute(-2, -3, -1, *range(s.dim() - 3))

        return s[0][range(batch_size), self.lens]
