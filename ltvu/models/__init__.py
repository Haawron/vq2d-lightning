from .vqloc import ClipMatcher as VQLoC
from .vqloc_prune_bca import ClipMatcher as VQLoC_prune_bca
from .vqloc_prune_aca import ClipMatcher as VQLoC_prune_aca

VQLoC_guide = VQLoC

__all__ = ['VQLoC', 'VQLoC_prune_bca', 'VQLoC_prune_aca', 'VQLoC_guide']
