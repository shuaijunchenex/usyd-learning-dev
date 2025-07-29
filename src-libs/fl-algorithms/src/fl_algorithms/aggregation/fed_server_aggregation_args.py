from __future__ import annotations
from aggregation.fed_server_aggregation_method import EFedServerAggregationMethod

'''
Dataset loader arguments
'''


class AggregatorArgs:
    """
    " Dataset loader arguments
    """

    def __init__(self):
        """
        Args for aggregation methods
        """
        self.aggregator_type: EFedServerAggregationMethod = EFedServerAggregationMethod.unknown

        #############################################################
        is_wbab: bool = False # for RBLA use
