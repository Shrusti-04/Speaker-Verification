"""
Models module initialization
"""

from .ecapa_tdnn import ECAPA_TDNN_Wrapper, AAMSoftmax
from .titanet import TiTANet_Wrapper

__all__ = ['ECAPA_TDNN_Wrapper', 'AAMSoftmax', 'TiTANet_Wrapper']
