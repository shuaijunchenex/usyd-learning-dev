from __future__ import annotations


class __global_const(object):
    """
    Const implementation
    Usage:
        Import first: 
           from usyd_learning.ml_utils import TextLogger, const
       
        Set constant value
           const.PI = 3.1415
    """

    class ConstError(TypeError): pass

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise self.ConstError("Changing const. %s" % key)
        else:
            self.__dict__[key] = value
        return

    def __getattr__(self, key):
        if self.__dict__.has_key(key):
            return self.key
        else:
            return None

import sys
sys.modules[__name__] = __global_const()
