'''
定义节点(Node)类型枚举
'''
class ENodeType(enumerate):    
    #未设置或未知
    Unknown = 0
    #服务器节点
    Server = 1
    #边节点
    Edge = 2
    #客户节点
    Client = 3