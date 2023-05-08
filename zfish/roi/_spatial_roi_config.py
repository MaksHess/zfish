import re
import warnings
from dataclasses import dataclass
from typing import Sequence


@dataclass
class SortKey:
    label_order: Sequence[str] = ('^emb.*$', '^cell.*$', '^nuc.*$', '^cyto.*$', '^mem.*$', '^loc.*$')
    dim_order: Sequence[str] = ('roi', 'l', 'c', 't', 'z', 'y', 'x')

    def labels(self, label_type: str) -> int:
        if self.label_order[0].startswith('^') and self.label_order[0].endswith('$'):
            try:
                return ['match' if re.match(p, label_type) else None for p in self.label_order].index('match')
            except ValueError as e:
                warnings.warn(f"`{label_type}` wasn't matched by any of {self.label_order}, returning index `1000`")
                return 1000
        else:
            return self.label_order.index(label_type)
        
    def images(self, channel_type: str) -> tuple[int, str]:
        try:
            stain = '-'.join(channel_type.split('-')[:-1])
            acquisition = int(channel_type.split('-')[-1])
            return acquisition, stain
        except ValueError as e:
            warnings.warn(f"`{channel_type}` must end in `-\\d+`, returning `(1000, '')`")
            return (1000, '')
        
    def dims(self, dim: str) -> int:
        return self.dim_order.index(dim)