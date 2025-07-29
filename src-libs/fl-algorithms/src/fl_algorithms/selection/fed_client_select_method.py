'''
' Node enumerate type
'''

from enum import StrEnum


class EFedClientSelectMethod(StrEnum):    

    """
    " Unknown
    """
    unknown = ""

    """
    " Random select clients
    """
    random = "random"

    """
    " Edge Node
    """
    high_loss = "high_loss"
    
    """
    " Client node
    """
    all = "all"
