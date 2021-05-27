# -*- coding: utf-8 -*-

import itertools
from functools import partial

import torch
from supar.structs import CRF2oDependency, CRFConstituency, CRFDependency
from supar.structs.semiring import LogSemiring, MaxSemiring, Semiring
from supar.utils.transform import CoNLL
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property


class BruteForceStructuredDistribution(Distribution):

    def __init__(self, scores, mask, **kwargs):
        self.mask = mask
        self.kwargs = kwargs

        self.scores = scores.requires_grad_() if isinstance(scores, torch.Tensor) else [s.requires_grad_() for s in scores]

    @lazy_property
    def log_partition(self):
        return torch.stack([LogSemiring.sum(i, -1) for i in self.enumerate(LogSemiring)])

    @lazy_property
    def max(self):
        return torch.stack([MaxSemiring.sum(i, -1) for i in self.enumerate(MaxSemiring)])

    @lazy_property
    def entropy(self):
        ps = [seq - self.log_partition[i] for i, seq in enumerate(self.enumerate(LogSemiring))]
        return -torch.stack([(i.exp() * i).sum() for i in ps])

    @lazy_property
    def count(self):
        structs = self.enumerate(Semiring)
        return torch.tensor([len(i) for i in structs]).to(structs[0].device).long()

    def cross_entropy(self, other):
        ps = [seq - self.log_partition[i] for i, seq in enumerate(self.enumerate(LogSemiring))]
        qs = [seq - other.log_partition[i] for i, seq in enumerate(other.enumerate(LogSemiring))]
        return -torch.stack([(i.exp() * j).sum() for i, j in zip(ps, qs)])

    def kl(self, other):
        return self.cross_entropy(other) - self.entropy

    def enumerate(self, semiring):
        raise NotImplementedError


class BruteForceCRFDependency(BruteForceStructuredDistribution):

    def __init__(self, scores, mask=None, multiroot=False, partial=False):
        super().__init__(scores, mask, multiroot=multiroot, partial=partial)

        self.multiroot = multiroot
        self.partial = partial

        self.mask = mask if mask is not None else scores.new_ones(scores.shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores.new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)

    def enumerate(self, semiring):
        trees = []
        for i, length in enumerate(self.mask.sum(-1).tolist()):
            trees.append([])
            for seq in itertools.product(range(length + 1), repeat=length):
                if not CoNLL.istree(list(seq), True, self.multiroot):
                    continue
                trees[-1].append(semiring.prod(self.scores[i, range(1, length + 1), seq], -1))
        return [torch.stack(seq) for seq in trees]


class BruteForceCRF2oDependency(BruteForceStructuredDistribution):

    def __init__(self, scores, mask=None, multiroot=False, partial=False):
        super().__init__(scores, mask, multiroot=multiroot, partial=partial)

        self.multiroot = multiroot
        self.partial = partial

        self.mask = mask if mask is not None else scores[0].new_ones(scores[0].shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores[0].new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)

    def enumerate(self, semiring):
        trees = []
        for i, length in enumerate(self.mask.sum(-1).tolist()):
            trees.append([])
            for seq in itertools.product(range(length + 1), repeat=length):
                if not CoNLL.istree(list(seq), True, self.multiroot):
                    continue
                sibs = self.lens.new_tensor(CoNLL.get_sibs(seq))
                sib_mask = sibs.gt(0)
                s_arc = semiring.prod(self.scores[0][i, range(1, length + 1), seq], -1)
                s_sib = semiring.prod(self.scores[1][i, 1:][sib_mask].gather(-1, sibs[sib_mask].unsqueeze(-1)).squeeze(-1))
                trees[-1].append(semiring.mul(s_arc, s_sib))
        return [torch.stack(seq) for seq in trees]


class BruteForceCRFConstituency(BruteForceStructuredDistribution):

    def __init__(self, scores, mask=None, labeled=False):
        super().__init__(scores, mask)

        self.labeled = labeled

        self.mask = mask if mask is not None else scores.new_ones(scores.shape[:(-1 if labeled else None)]).bool().triu_(1)
        self.lens = self.mask[:, 0].sum(-1)

    def enumerate(self, semiring):
        scores = self.scores if self.labeled else self.scores.unsqueeze(-1)

        def enumerate(s, i, j):
            if i + 1 == j:
                yield from s[i, j].unbind(-1)
            for k in range(i + 1, j):
                for t1 in enumerate(s, i, k):
                    for t2 in enumerate(s, k, j):
                        for t in s[i, j].unbind(-1):
                            yield semiring.times(t, t1, t2)
        return [torch.stack([i for i in enumerate(s, 0, length)]) for s, length in zip(scores, self.lens)]


def test_struct():
    torch.manual_seed(1)
    batch_size, seq_len = 2, 5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def enumerate():
        structs = [
            (CRFDependency, BruteForceCRFDependency),
            (partial(CRFDependency, multiroot=True), partial(BruteForceCRFDependency, multiroot=True)),
            (CRF2oDependency, BruteForceCRF2oDependency),
            (partial(CRF2oDependency, multiroot=True), partial(BruteForceCRF2oDependency, multiroot=True)),
            (CRFConstituency, BruteForceCRFConstituency),
            (partial(CRFConstituency, labeled=True), partial(BruteForceCRFConstituency, labeled=True)),
        ]
        for struct, brute_force in structs:
            for _ in range(5):
                if struct == CRF2oDependency or (isinstance(struct, partial) and struct.func == CRF2oDependency):
                    s1 = [torch.randn(batch_size, seq_len, seq_len).to(device),
                          torch.randn(batch_size, seq_len, seq_len, seq_len).to(device)]
                    s2 = [torch.randn(batch_size, seq_len, seq_len).to(device),
                          torch.randn(batch_size, seq_len, seq_len, seq_len).to(device)]
                elif isinstance(struct, partial) and struct.func == CRFConstituency and struct.keywords['labeled']:
                    s1 = torch.randn(batch_size, seq_len, seq_len, 2).to(device).fill_(0)
                    s2 = torch.randn(batch_size, seq_len, seq_len, 2).to(device).fill_(0)
                else:
                    s1 = torch.randn(batch_size, seq_len, seq_len).to(device)
                    s2 = torch.randn(batch_size, seq_len, seq_len).to(device)
                yield struct(s1), struct(s2), brute_force(s1), brute_force(s2)

    for s1, s2, b1, b2 in enumerate():
        assert s1.max.allclose(b1.max)
        assert s1.log_partition.allclose(b1.log_partition)
        assert s1.entropy.allclose(b1.entropy)
        assert s1.count.allclose(b1.count)
        assert s1.cross_entropy(s2).allclose(b1.cross_entropy(b2))
        assert s1.kl(s2).allclose(b1.kl(b2))
