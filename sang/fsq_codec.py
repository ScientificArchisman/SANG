"""Mixed-radix decomposition of VidTok FSQ token indices from the implicit codebook."""
import math

import torch


class FSQIndexCodec:
    """Bidirectional map between FSQ token indices and per-dimension digits."""

    def __init__(self, levels, digits_lut, index_lut):
        self.levels = tuple(int(x) for x in levels)
        self.n_dims = len(self.levels)
        self.vocab = int(index_lut.numel())
        self._digits_lut = digits_lut
        self._index_lut = index_lut

    @classmethod
    def from_codebook(cls, C: torch.Tensor):
        """C: [V, n_dims] FSQ implicit codebook."""
        C = C.detach().cpu()
        V, n = C.shape
        levels, digit_cols = [], []
        for d in range(n):
            vals = torch.unique(C[:, d])
            levels.append(vals.numel())
            digit_cols.append(torch.searchsorted(vals, C[:, d].contiguous()))
        digits = torch.stack(digit_cols, dim=-1).long()

        prod = 1
        for l in levels:
            prod *= l
        if prod != V:
            raise ValueError(f"prod(levels)={prod} != V={V}; not a clean FSQ product lattice")

        index_lut = torch.full(tuple(levels), -1, dtype=torch.long)
        index_lut[tuple(digits.T)] = torch.arange(V, dtype=torch.long)
        if (index_lut < 0).any():
            raise ValueError("digit tuples are not a bijection onto [0, V)")
        return cls(levels, digits, index_lut)

    def to(self, device):
        self._digits_lut = self._digits_lut.to(device)
        self._index_lut = self._index_lut.to(device)
        return self

    def to_digits(self, idx: torch.Tensor) -> torch.Tensor:
        return self._digits_lut[idx.long()]

    def from_digits(self, digits: torch.Tensor) -> torch.Tensor:
        return self._index_lut[tuple(digits.long().movedim(-1, 0))]

    def self_check(self, verbose: bool = True) -> bool:
        idx = torch.arange(self.vocab, device=self._digits_lut.device)
        ok = bool(torch.equal(self.from_digits(self.to_digits(idx)), idx))
        if verbose:
            ent = sum(math.log(l) for l in self.levels)
            print(f"[FSQIndexCodec] V={self.vocab} levels={self.levels} "
                  f"roundtrip={ok} sum(ln levels)={ent:.4f} ln(V)={math.log(self.vocab):.4f}")
        if not ok:
            raise RuntimeError("FSQIndexCodec round-trip failed")
        return ok
