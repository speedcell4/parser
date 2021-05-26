# -*- coding: utf-8 -*-

import itertools
from functools import partial

import torch
from supar.structs import CRF2oDependency, CRFDependency
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
        seqs = self.enumerate(Semiring)
        return torch.tensor([len(i) for i in seqs]).to(seqs[0].device).long()

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
        seqs = []
        for i, length in enumerate(self.mask.sum(-1).tolist()):
            seqs.append([])
            for seq in itertools.product(range(length + 1), repeat=length):
                if not CoNLL.istree(list(seq), True, self.multiroot):
                    continue
                seqs[-1].append(semiring.prod(self.scores[i, range(1, length + 1), seq], -1))
        return [torch.stack(seq) for seq in seqs]


class BruteForceCRF2oDependency(BruteForceStructuredDistribution):

    def __init__(self, scores, mask=None, multiroot=False, partial=False):
        super().__init__(scores, mask, multiroot=multiroot, partial=partial)

        self.multiroot = multiroot
        self.partial = partial

        self.mask = mask if mask is not None else scores[0].new_ones(scores[0].shape[:2]).bool()
        self.mask = self.mask.index_fill(1, scores[0].new_tensor(0).long(), 0)
        self.lens = self.mask.sum(-1)

    def enumerate(self, semiring):
        seqs = []
        for i, length in enumerate(self.mask.sum(-1).tolist()):
            seqs.append([])
            for seq in itertools.product(range(length + 1), repeat=length):
                if not CoNLL.istree(list(seq), True, self.multiroot):
                    continue
                sibs = self.lens.new_tensor(CoNLL.get_sibs(seq))
                sib_mask = sibs.gt(0)
                s_arc = semiring.prod(self.scores[0][i, range(1, length + 1), seq], -1)
                s_sib = semiring.prod(self.scores[1][i, 1:][sib_mask].gather(-1, sibs[sib_mask].unsqueeze(-1)).squeeze(-1))
                seqs[-1].append(semiring.mul(s_arc, s_sib))
        return [torch.stack(seq) for seq in seqs]


def test_struct():
    torch.manual_seed(1)
    batch_size, seq_len = 2, 5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    structs = [(CRFDependency, BruteForceCRFDependency, False),
               (partial(CRFDependency, multiroot=True), partial(BruteForceCRFDependency, multiroot=True), False),
               (CRF2oDependency, BruteForceCRF2oDependency, True),
               (partial(CRF2oDependency, multiroot=True), partial(BruteForceCRF2oDependency, multiroot=True), True)]
    for struct, brute_force, second_order in structs:
        for i in range(5):
            s1, s2 = torch.randn(batch_size, seq_len, seq_len).to(device), torch.randn(batch_size, seq_len, seq_len).to(device)
            if second_order:
                s1 = [s1, torch.randn(batch_size, seq_len, seq_len, seq_len).to(device)]
                s2 = [s2, torch.randn(batch_size, seq_len, seq_len, seq_len).to(device)]
            struct1, struct2 = struct(s1), struct(s2)
            brute_force1, brute_force2 = brute_force(s1), brute_force(s2)
            assert struct1.max.allclose(brute_force1.max)
            assert struct1.log_partition.allclose(brute_force1.log_partition)
            assert struct1.entropy.allclose(brute_force1.entropy)
            assert struct1.count.allclose(brute_force1.count)
            assert struct1.cross_entropy(struct2).allclose(brute_force1.cross_entropy(brute_force2))
            assert struct1.kl(struct2).allclose(brute_force1.kl(brute_force2))
