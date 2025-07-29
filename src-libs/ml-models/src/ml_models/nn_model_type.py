"""
Define NN Model type enum
"""

class ENNModelType(enumerate):
    unknown = 0
    
    mnist2NNBrenden = 1
    capstoneMLP = 2
    simpleMLP = 3
    cifarConvnet = 4

    simpleLoRAMLP = 5