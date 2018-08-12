"""
Interpreter
-----------

AST-walking interpreter.
"""
from __future__ import print_function
import operator
from collections import namedtuple
import CREAM.ast as ast
from CREAM.lexer import Lexer, TokenStream
from CREAM.parser import Parser
from CREAM.errors import aSyntaxError, report_syntax_error
from CREAM.utils import print_ast, print_tokens, print_env


BuiltinFunction = namedtuple('BuiltinFunction', ['params', 'body'])


class Break(Exception):
    pass


class Continue(Exception):
    pass


class Return(Exception):
    def __init__(self, value):
        self.value = value


class Environment(object):

    def __init__(self, parent=None, args=None):
        self._parent = parent
        self._values = {}
        if args is not None:
            self._from_dict(args)

    def _from_dict(self, args):
        for key, value in args.items():
            self.set(key, value)

    def set(self, key, val):
        self._values[key] = val

    def get(self, key):
        val = self._values.get(key, None)
        if val is None and self._parent is not None:
            return self._parent.get(key)
        else:
            return val

    def asdict(self):
        return self._values

    def __repr__(self):
        return 'Environment({})'.format(str(self._values))


def eval_binary_operator(node, env):
    simple_operations = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '%': operator.mod,
        '>': operator.gt,
        '>=': operator.ge,
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        'is': operator.is_,
        '..': range,
        '...': lambda start, end: range(start, end + 1),
    }
    lazy_operations = {
        '&&': lambda lnode, lenv: bool(eval_expression(lnode.left, lenv)) and bool(eval_expression(lnode.right, lenv)),
        '||': lambda lnode, lenv: bool(eval_expression(lnode.left, lenv)) or bool(eval_expression(lnode.right, lenv)),
    }

    if node.operator in simple_operations:
        return simple_operations[node.operator](eval_expression(node.left, env), eval_expression(node.right, env))
    elif node.operator in lazy_operations:
        return lazy_operations[node.operator](node, env)
    else:
        raise Exception('Invalid operator {}'.format(node.operator))


def eval_unary_operator(node, env):
    operations = {
        '-': operator.neg,
        '!': operator.not_,
    }
    return operations[node.operator](eval_expression(node.right, env))


def eval_assignment(node, env):
    if isinstance(node.left, ast.SubscriptOperator):
        return eval_setitem(node, env)
    else:
        return env.set(node.left.value, eval_expression(node.right, env))


def eval_operation_assignment(node, env):
    operations = {
        '+=': operator.add,
        '-=': operator.sub,
        '*=': operator.mul,
        '/=': operator.truediv,
        '%=': operator.mod,
    }
    if node.assignment in operations:
        result = operations[node.assignment](eval_expression(node.left, env), eval_expression(node.right, env))
        return env.set(node.left.value, result)
    else:
        raise Exception('Invalid operator {}'.format(node.assignment))


def eval_condition(node, env):
    if eval_expression(node.test, env):
        return eval_statements(node.if_body, env)

    for cond in node.elifs:
        if eval_expression(cond.test, env):
            return eval_statements(cond.body, env)

    if node.else_body is not None:
        return eval_statements(node.else_body, env)


def eval_match(node, env):
    test = eval_expression(node.test, env)
    for pattern in node.patterns:
        if eval_expression(pattern.pattern, env) == test:
            return eval_statements(pattern.body, env)
    if node.else_body is not None:
        return eval_statements(node.else_body, env)


def eval_while_op_loop(node, env):
    while eval_expression(node.test, env):
        try:
            eval_statements(node.body, env)
        except Break:
            break
        except Continue:
            pass


def eval_while_in_loop(node, env):
    var_name = node.var_name
    collection = eval_expression(node.collection, env)
    for val in collection:
        env.set(var_name, val)
        try:
            eval_statements(node.body, env)
        except Break:
            break
        except Continue:
            pass


def eval_entity_declaration(node, env):
    return env.set(node.name, node)


def eval_function_declaration(node, env):
    return env.set(node.name, node)


def eval_entity_init(node, env):
    entity_members = Environment()
    entity = eval_expression(node.left, env)
    init_func = list(filter(lambda x: isinstance(x, ast.Function) and x.name == 'init', entity.body))[0]
    n_expected_args = len(init_func.params)
    n_actual_args = len(node.arguments)

    if n_expected_args != n_actual_args:
        raise TypeError('Expected {} arguments, got {}'.format(n_expected_args, n_actual_args))

    args = dict(zip(init_func.params, [eval_expression(node, env) for node in node.arguments]))
    call_env = Environment(env, args)

    for m in entity.body:
        if isinstance(m, ast.Assignment):
            entity_members.set(m.left, m.right)
        elif isinstance(m, ast.Function):
            entity_members.set(m.name, m)

    init_members = entity_members
    add_builtins(init_members)

    for k in call_env._values:
        init_members.set(k, call_env.get(k))

    eval_statements(init_func.body, init_members)
    return entity_members


def eval_member_access(node, env):
    if isinstance(node.right, ast.Call):
        return eval_call(node, env)
    elif isinstance(node.right, ast.Identifier):
        return eval_identifier(node, env)


def eval_self_reference(node, env):
    if isinstance(node.right, ast.Call):
        return eval_call(node.right, env)
    elif isinstance(node.right, ast.Assignment):
        return eval_assignment(node.right, env)


def eval_call(node, env):
    function = None
    n_actual_args = None
    args = None

    if not isinstance(node, ast.MemberAccess):
        function = eval_expression(node.left, env)
        args = node.arguments
    else:
        function = env.get(node.left.value)
        args = node.right.arguments

    if isinstance(function, ast.Entity):
        return eval_entity_init(node, env)

    n_expected_args = len(function.params)
    n_actual_args = len(args)

    if n_expected_args != n_actual_args:
        raise TypeError('Expected {} arguments, got {}'.format(n_expected_args, n_actual_args))

    args = dict(zip(function.params, [eval_expression(node, env) for node in args]))
    if isinstance(function, BuiltinFunction):
        return function.body(args, env)
    else:
        call_env = Environment(env, args)
        try:
            return eval_statements(function.body, call_env)
        except Return as ret:
            return ret.value


def eval_identifier(node, env):
    name = None
    val = None

    if isinstance(node, ast.Identifier):
        name = node.value
        val = env.get(name)
    elif isinstance(node, ast.MemberAccess):
        name = node.right
        val = env.get(node.left.value)[name].value

    if val is None:
        raise NameError('Name "{}" is not defined'.format(name))
    return val


def eval_getitem(node, env):
    collection = eval_expression(node.left, env)
    key = eval_expression(node.key, env)
    return collection[key]


def eval_setitem(node, env):
    collection = eval_expression(node.left.left, env)
    key = eval_expression(node.left.key, env)
    collection[key] = eval_expression(node.right, env)


def eval_array(node, env):
    return [eval_expression(item, env) for item in node.items]


def eval_dict(node, env):
    return {eval_expression(key, env): eval_expression(value, env) for key, value in node.items}


def eval_return(node, env):
    return eval_expression(node.value, env) if node.value is not None else None


evaluators = {
    ast.Array: eval_array,
    ast.Assignment: eval_assignment,
    ast.BinaryOperator: eval_binary_operator,
    ast.Call: eval_call,
    ast.Condition: eval_condition,
    ast.Dictionary: eval_dict,
    ast.Entity: eval_entity_declaration,
    ast.Function: eval_function_declaration,
    ast.Identifier: eval_identifier,
    ast.MemberAccess: eval_member_access,
    ast.Number: lambda node, env: node.value,
    ast.OperationAssignment: eval_operation_assignment,
    ast.Return: eval_return,
    ast.SelfReference: eval_self_reference,
    ast.String: lambda node, env: node.value,
    ast.SubscriptOperator: eval_getitem,
    ast.UnaryOperator: eval_unary_operator,
    ast.WhileOPLoop: eval_while_op_loop,
    ast.WhileINLoop: eval_while_in_loop,
    ast.Match: eval_match,
}


def eval_node(node, env):
    tp = type(node)
    if tp in evaluators:
        return evaluators[tp](node, env)
    else:
        raise Exception('Unknown node {} {}'.format(tp.__name__, node))


def eval_expression(node, env):
    return eval_node(node, env)


def eval_statement(node, env):
    return eval_node(node, env)


def eval_statements(statements, env):
    ret = None
    for statement in statements:
        if isinstance(statement, ast.Exit):
            raise Break(ret)
        elif isinstance(statement, ast.Next):
            raise Continue(ret)

        ret = eval_statement(statement, env)
        if isinstance(statement, ast.Return):
            raise Return(ret)
    return ret


def add_builtins(env):
    builtins = {
        'print': (['value'], lambda args, e: print(args['value'])),
        'len': (['iter'], lambda args, e: len(args['iter'])),
        'slice': (['iter', 'start', 'stop'], lambda args, e: list(args['iter'][args['start']:args['stop']])),
        'str': (['in'], lambda args, e: str(args['in'])),
        'int': (['in'], lambda args, e: int(args['in'])),
    }
    for key, (params, func) in builtins.items():
        env.set(key, BuiltinFunction(params, func))


def create_global_env():
    env = Environment()
    add_builtins(env)
    return env


def evaluate_env(s, env, trace=False):
    lexer = Lexer()
    try:
        tokens = lexer.tokenize(s)
    except aSyntaxError as err:
        report_syntax_error(lexer, err)
        if trace:
            raise
        else:
            return

    if trace:
        print('Tokens')
        print_tokens(tokens)
        print()

    token_stream = TokenStream(tokens)

    try:
        program = Parser().parse(token_stream)
    except aSyntaxError as err:
        report_syntax_error(lexer, err)
        if trace:
            raise
        else:
            return

    if trace:
        print('AST')
        print_ast(program.body)
        print()

    ret = eval_statements(program.body, env)

    if trace:
        print('Environment')
        print_env(env)
        print()

    return ret


def evaluate(s, trace=False):
    return evaluate_env(s, create_global_env(), trace)
