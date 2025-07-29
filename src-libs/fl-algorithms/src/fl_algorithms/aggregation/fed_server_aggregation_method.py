'''
' Fed Server Aggregation enumerate type
'''

from enum import StrEnum


class EFedServerAggregationMethod(StrEnum):    

    """
    " Unknown
    """
    unknown = ""

    """
    " fedavg
    """
    fedavg = "fedavg"
 
    """
    " rbla
    """
    rbla = "rbla"

    """
    " flexlora
    """
    flexlora = "flexlora"
