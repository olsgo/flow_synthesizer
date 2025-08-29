import re
from typing import List, Set

# Conservative list of clearly binary parameters in PolyMAX
_BINARY_PARAMS: Set[str] = {
    'arp_enable',
    'lfo_sync',
    'osc_2_sync',
    'mod_fx_enable',
    'master_bypass',
}

def get_binary_param_indices(param_names: List[str]) -> List[int]:
    """Return indices of parameters that are binary toggles.

    The list is intentionally conservative to avoid misclassification.
    """
    idx = []
    for i, name in enumerate(param_names):
        if name in _BINARY_PARAMS:
            idx.append(i)
    return idx

