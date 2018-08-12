"""
AST
---

Abstract syntax tree nodes.
"""

from collections import namedtuple

Array = namedtuple('Array', ['items'])
Assignment = namedtuple('Assignment', ['left', 'right'])
BinaryOperator = namedtuple('BinaryOperator', ['operator', 'left', 'right'])
Call = namedtuple('Call', ['left', 'arguments'])
Condition = namedtuple('Condition', ['test', 'if_body', 'elifs', 'else_body'])
ConditionElif = namedtuple('ConditionElif', ['test', 'body'])
Dictionary = namedtuple('Dictionary', ['items'])
Entity = namedtuple('Entity', ['name', 'params', 'body'])
Exit = namedtuple('Exit', [])
Function = namedtuple('Function', ['name', 'params', 'body'])
Identifier = namedtuple('Identifier', ['value'])
MemberAccess = namedtuple('Member', ['left', 'right'])
Next = namedtuple('Next', [])
Number = namedtuple('Number', ['value'])
OperationAssignment = namedtuple('Assignment', ['left', 'assignment', 'right'])
Program = namedtuple('Program', ['body'])
Return = namedtuple('Return', ['value'])
SelfReference = namedtuple('Self', ['right', 'ref_name'])
String = namedtuple('String', ['value'])
SubscriptOperator = namedtuple('SubscriptOperator', ['left', 'key'])
UnaryOperator = namedtuple('UnaryOperator', ['operator', 'right'])
WhileINLoop = namedtuple('WhileLoop', ['var_name', 'collection', 'body'])
WhileOPLoop = namedtuple('WhileLoop', ['test', 'body'])


Match = namedtuple('Match', ['test', 'patterns', 'else_body'])
MatchPattern = namedtuple('MatchPattern', ['pattern', 'body'])
