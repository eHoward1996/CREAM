"""
Parser
------

Top-down recursive descent parser.
"""

import CREAM.ast as ast
from CREAM.errors import aSyntaxError


class ParserError(aSyntaxError):

    def __init__(self, message, token):
        super(ParserError, self).__init__(message, token.line, token.column)


def enter_scope(parser, name):
    class State(object):
        def __enter__(self):
            parser.scope.append(name)

        def __exit__(self, exc_type, exc_val, exc_tb):
            parser.scope.pop()

    return State()


class Subparser(object):

    PRECEDENCE = {
        'call': 10,
        'subscript': 10,
        '.': 10,

        'unary': 9,

        '*': 7,
        '/': 7,
        '%': 7,

        '+': 6,
        '-': 6,

        '>': 5,
        '>=': 5,
        '<': 5,
        '<=': 5,

        '==': 4,
        '!=': 4,

        '&&': 3,

        '||': 2,

        '*=': 1,
        '/=': 1,
        '%=': 1,
        '+=': 1,
        '-=': 1,

        '..': 1,
        '...': 1,
    }

    def get_subparser(self, token, subparsers, default=None):
        cls = subparsers.get(token.name, default)
        if cls is not None:
            return cls()


# entity_body: LBRACE NEWLINE INDENT stmts DEDENT RBRACE NEWLINE
class EntityBody(Subparser):
    def get_entity_subparser(self, token):
        return self.get_subparser(token, {
            'FUNCTION': FunctionStatement,
            # 'IF': ConditionalStatement,
        }, ExpressionStatement)

    def parse(self, parser, tokens):
        tokens.consume_expected('LBRACE', 'NEWLINE')
        statements = []

        # If this is true, this is a proper block with statements
        # otherwise, this is probably an empty block
        if tokens.current().name == 'INDENT':
            tokens.consume_expected('INDENT')
            while not tokens.is_end():
                statement = self.get_entity_subparser(
                    tokens.current()).parse(parser, tokens)
                if statement is not None:
                    statements.append(statement)
                else:
                    break
            tokens.consume_expected('DEDENT', 'RBRACE', 'NEWLINE')
        else:
            tokens.consume_expected('RBRACE', 'NEWLINE')
        return statements


# entity_stmnt: ENTITY NAME (LARROW LPAREN entity_params? RPAREN)? { entity block }
class EntityStatement(Subparser):

    def _parse_parents(self, tokens):
        parents = []
        if tokens.current().name == 'NAME':
            while not tokens.is_end():
                id_token = tokens.consume_expected('NAME')
                parents.append(id_token.value)

                if tokens.current().name == 'COMMA':
                    tokens.consume_expected('COMMA')
                else:
                    break
        return parents

    def parse(self, parser, tokens):
        parents = []
        tokens.consume_expected('ENTITY')
        id_token = tokens.consume_expected('NAME')

        if tokens.current().name == 'LARROW':
            tokens.consume_expected('LARROW', 'LPAREN')
            parents = self._parse_parents(tokens)
            tokens.consume_expected('RPAREN')

        with enter_scope(parser, ('entity', id_token.value)):
            body = EntityBody().parse(parser, tokens)
        if body is None:
            raise ParserError('Expected entity body', tokens.current())
        return ast.Entity(id_token.value, parents, body)


# self_ref: SELF
class SelfReferenceStatement(Subparser):

    def parse(self, parser, tokens):
        scope = [item for item in parser.scope if item[0] == 'entity']
        if scope is None or isinstance(scope, tuple):
            raise ParserError('Self Reference cannot exist outside of an entity', tokens.current())

        tokens.consume_expected('SELF', 'MEMBER')
        right = ExpressionStatement().parse(parser, tokens)
        return ast.SelfReference(right, scope[0][1])


# block: LBRACE NEWLINE INDENT stmnts DEDENT RBRACE NEWLINE
class Block(Subparser):

    def parse(self, parser, tokens):
        tokens.consume_expected('LBRACE', 'NEWLINE')
        statements = []
        # If this is true, this is a proper block with statements
        # otherwise, this is probably an empty block
        if tokens.current().name == 'INDENT':
            tokens.consume_expected('INDENT')
            statements = Statements().parse(parser, tokens)
            tokens.consume_expected('DEDENT')

        tokens.consume_expected('RBRACE', 'NEWLINE')
        return statements


# func_stmnt: FUNCTION NAME LPAREN func_params? RPAREN block
class FunctionStatement(Subparser):

    # func_params: (NAME COMMA)*
    def _parse_params(self, tokens):
        params = []
        if tokens.current().name == 'NAME':
            while not tokens.is_end():
                id_token = tokens.consume_expected('NAME')
                params.append(id_token.value)
                if tokens.current().name == 'COMMA':
                    tokens.consume_expected('COMMA')
                else:
                    break
        return params

    def parse(self, parser, tokens):
        tokens.consume_expected('FUNCTION')
        id_token = tokens.consume_expected('NAME')
        tokens.consume_expected('RARROW', 'LPAREN')
        arguments = self._parse_params(tokens)
        tokens.consume_expected('RPAREN')

        with enter_scope(parser, ('function', id_token.value)):
            block = Block().parse(parser, tokens)

        if block is None:
            raise ParserError('Expected function body', tokens.current())
        return ast.Function(id_token.value, arguments, block)


# cond_stmnt: expr ? { block } (expr ?? { block })* (?! { block })?
class ConditionalStatement(Subparser):

    def _parse_elif_conditions(self, parser, tokens):
        conditions = []
        while not tokens.is_end() and tokens.current().name == 'ELIF':
            # Consume initial block identifier
            tokens.consume_expected('ELIF')

            # Begin boolean expression
            tokens.consume_expected('LPAREN')
            test = Expression().parse(parser, tokens)
            if test is None:
                raise ParserError('Expected `elif` condition', tokens.current())

            tokens.consume_expected('RPAREN')

            # Make sure ELIF is ended properly
            tokens.consume_expected('ELIF')

            with enter_scope(parser, 'elif_cond'):
                block = Block().parse(parser, tokens)
            if block is None:
                raise ParserError('Expected `elif` body', tokens.current())
            conditions.append(ast.ConditionElif(test, block))

        return conditions

    def _parse_else(self, parser, tokens):
        block = None
        if not tokens.is_end() and tokens.current().name == 'ELSE':
            # Consume intitial block identifier
            tokens.consume_expected('ELSE')
            with enter_scope(parser, 'else_cond'):
                block = Block().parse(parser, tokens)
            if block is None:
                raise ParserError('Expected `else` body', tokens.current())
        return block

    def parse(self, parser, tokens):
        # Consume initial block identifier
        tokens.consume_expected('IF')

        # Begin boolean expression
        tokens.consume_expected('LPAREN')
        test = Expression().parse(parser, tokens)

        # Check that an expression was entered
        if test is None:
            raise ParserError('Expected `if` condition', tokens.current())

        tokens.consume_expected('RPAREN')

        # Make sure the statement is ended properly
        tokens.consume_expected('IF')
        with enter_scope(parser, 'if_cond'):
            block = Block().parse(parser, tokens)
        if block is None:
            raise ParserError('Expected if body', tokens.current())

        elif_conditions = self._parse_elif_conditions(parser, tokens)
        else_block = self._parse_else(parser, tokens)
        return ast.Condition(test, block, elif_conditions, else_block)


# match_stmnt: MATCH expr COLON NEWLINE INDENT match_when+ (ELSE COLON block)? DEDENT
class MatchStatement(Subparser):

    # match_when: WHEN expr COLON block
    def _parse_when(self, parser, tokens):
        tokens.consume_expected('WHEN')
        pattern = Expression().parse(parser, tokens)
        if pattern is None:
            raise ParserError('Pattern expression expected', tokens.current())
        tokens.consume_expected('COLON')
        block = Block().parse(parser, tokens)
        return ast.MatchPattern(pattern, block)

    def parse(self, parser, tokens):
        tokens.consume_expected('MATCH')
        test = Expression().parse(parser, tokens)
        tokens.consume_expected('COLON', 'NEWLINE', 'INDENT')
        patterns = []
        while not tokens.is_end() and tokens.current().name == 'WHEN':
            patterns.append(self._parse_when(parser, tokens))
        if not patterns:
            raise ParserError('One or more `when` pattern excepted', tokens.current())
        else_block = None
        if not tokens.is_end() and tokens.current().name == 'ELSE':
            tokens.consume_expected('ELSE', 'COLON')
            else_block = Block().parse(parser, tokens)
            if else_block is None:
                raise ParserError('Expected `else` body', tokens.current())
        tokens.consume_expected('DEDENT')
        return ast.Match(test, patterns, else_block)


# loop_while_stmnt: WHILE expr { block }
class WhileLoopStatement(Subparser):

    def parse(self, parser, tokens):
        op_test = None
        collection = None
        tokens.consume_expected('WHILE', 'LPAREN')

        # Check if this is a While...IN statement
        if tokens.next().name == 'IN':
            id_token = tokens.consume_expected('NAME')
            tokens.consume_expected('IN')
            collection = Expression().parse(parser, tokens)
        else:
            op_test = Expression().parse(parser, tokens)

        tokens.consume_expected('RPAREN')
        if op_test is None and collection is None:
            raise ParserError('While condition expected', tokens.current())
        with enter_scope(parser, 'loop'):
            block = Block().parse(parser, tokens)
        if block is None:
            raise ParserError('Expected loop body', tokens.current())

        if op_test is not None:
            return ast.WhileOPLoop(op_test, block)
        else:
            return ast.WhileINLoop(id_token.value, collection, block)


# return_stmnt: RETURN expr?
class ReturnStatement(Subparser):

    def parse(self, parser, tokens):
        scope = [item for item in parser.scope if item[0] == 'function']
        if scope is None or isinstance(scope, tuple):
            raise ParserError('Return outside of function', tokens.current())
        tokens.consume_expected('RETURN')
        value = Expression().parse(parser, tokens)
        tokens.consume_expected('NEWLINE')
        return ast.Return(value)


# break_stmnt: EXIT
class ExitStatement(Subparser):

    def parse(self, parser, tokens):
        if not parser.scope or parser.scope[-1] != 'loop':
            raise ParserError('Exit outside of loop', tokens.current())
        tokens.consume_expected('EXIT', 'NEWLINE')
        return ast.Exit()


# cont_stmnt: NEXT
class NextStatement(Subparser):

    def parse(self, parser, tokens):
        if not parser.scope or parser.scope[-1] != 'loop':
            raise ParserError('Next outside of loop', tokens.current())
        tokens.consume_expected('NEXT', 'NEWLINE')
        return ast.Next()


# operation_assign: expr OPER ASSIGN expr NEWLINE
class OperationAssignmentStatement(Subparser):

    def parse(self, parser, tokens, left):
        assignment = tokens.consume_expected('OPASSIGN').value
        right = Expression().parse(parser, tokens)
        tokens.consume_expected('NEWLINE')
        return ast.OperationAssignment(left, assignment, right)


# member_access: expr MEMBER_OP expr
class MemberAccessStatement(Subparser):

    def parse(self, parser, tokens, left):
        tokens.consume_expected('MEMBER')
        right = Expression().parse(parser, tokens)
        return ast.MemberAccess(left, right)


# assing_stmnt: expr ASSIGN expr NEWLINE
class AssignmentStatement(Subparser):

    def parse(self, parser, tokens, left):
        tokens.consume_expected('ASSIGN')
        right = Expression().parse(parser, tokens)
        tokens.consume_expected('NEWLINE')
        return ast.Assignment(left, right)


class PrefixSubparser(Subparser):

    def parse(self, parser, tokens):
        raise NotImplementedError()


class InfixSubparser(Subparser):

    def parse(self, parser, tokens, left):
        raise NotImplementedError()

    def get_precedence(self, token):
        raise NotImplementedError()


# number_expr: NUMBER
class NumberExpression(PrefixSubparser):

    def parse(self, parser, tokens):
        token = tokens.consume_expected('NUMBER')
        return ast.Number(token.value)


# str_expr: STRING
class StringExpression(PrefixSubparser):

    def parse(self, parser, tokens):
        token = tokens.consume_expected('STRING')
        return ast.String(token.value)


# name_expr: NAME
class NameExpression(PrefixSubparser):

    def parse(self, parser, tokens):
        token = tokens.consume_expected('NAME')
        if tokens.current().name == 'MEMBER':
            return MemberAccessStatement().parse(parser, tokens, token)
        return ast.Identifier(token.value)


# prefix_expr: OPERATOR expr
class UnaryOperatorExpression(PrefixSubparser):

    SUPPORTED_OPERATORS = ['-', '!']

    def parse(self, parser, tokens):
        token = tokens.consume_expected('OPERATOR')
        if token.value not in self.SUPPORTED_OPERATORS:
            raise ParserError(
                'Unary operator {} is not supported'.format(token.value), token)
        right = Expression().parse(parser, tokens, self.get_precedence(token))
        if right is None:
            raise ParserError('Expected expression'.format(
                token.value), tokens.consume())
        return ast.UnaryOperator(token.value, right)

    def get_precedence(self, token):
        return self.PRECEDENCE['unary']


# group_expr: LPAREN expr RPAREN
class GroupExpression(PrefixSubparser):

    def parse(self, parser, tokens):
        tokens.consume_expected('LPAREN')
        right = Expression().parse(parser, tokens)
        tokens.consume_expected('RPAREN')
        return right


# list_of_expr: (expr COMMA)*
class ListOfExpressions(Subparser):

    def parse(self, parser, tokens):
        items = []
        while not tokens.is_end():
            exp = Expression().parse(parser, tokens)
            if exp is not None:
                items.append(exp)
            else:
                break
            if tokens.current().name == 'COMMA':
                tokens.consume_expected('COMMA')
            else:
                break
        return items


# array_expr: LBRACK list_of_expr? RBRACK
class ArrayExpression(PrefixSubparser):

    def parse(self, parser, tokens):
        tokens.consume_expected('LBRACK')
        items = ListOfExpressions().parse(parser, tokens)
        tokens.consume_expected('RBRACK')
        return ast.Array(items)


# dict_expr: LBRACE (expr COLON expr COMMA)* RBRACE
class DictionaryExpression(PrefixSubparser):

    def _parse_keyvals(self, parser, tokens):
        items = []
        while not tokens.is_end():
            key = Expression().parse(parser, tokens)
            if key is not None:
                tokens.consume_expected('COLON')
                value = Expression().parse(parser, tokens)
                if value is None:
                    raise ParserError(
                        'Dictionary value expected', tokens.consume())
                items.append((key, value))
            else:
                break
            if tokens.current().name == 'COMMA':
                tokens.consume_expected('COMMA')
            else:
                break
        return items

    def parse(self, parser, tokens):
        tokens.consume_expected('LCBRACK')
        items = self._parse_keyvals(parser, tokens)
        tokens.consume_expected('RCBRACK')
        return ast.Dictionary(items)


# infix_expr: expr OPERATOR expr
class BinaryOperatorExpression(InfixSubparser):

    def parse(self, parser, tokens, left):
        token = tokens.consume_expected('OPERATOR')
        right = Expression().parse(parser, tokens, self.get_precedence(token))

        if right is None:
            raise ParserError('Expected expression'.format(token.value), tokens.consume())
        return ast.BinaryOperator(token.value, left, right)

    def get_precedence(self, token):
        return self.PRECEDENCE[token.value]


# call_expr: NAME LPAREN list_of_expr? RPAREN
class CallExpression(InfixSubparser):

    def parse(self, parser, tokens, left):
        tokens.consume_expected('LPAREN')
        arguments = ListOfExpressions().parse(parser, tokens)
        tokens.consume_expected('RPAREN')
        return ast.Call(left, arguments)

    def get_precedence(self, token):
        return self.PRECEDENCE['call']


# subscript_expr: NAME [ expr ]
class SubscriptOperatorExpression(InfixSubparser):

    def parse(self, parser, tokens, left):
        tokens.consume_expected('LBRACK')
        key = Expression().parse(parser, tokens)
        if key is None:
            raise ParserError(
                'Subscript operator key is required', tokens.current())
        tokens.consume_expected('RBRACK')
        return ast.SubscriptOperator(left, key)

    def get_precedence(self, token):
        return self.PRECEDENCE['subscript']


# expr: number_expr | str_expr | name_expr | group_expr | array_expr | dict_expr | prefix_expr | infix_expr | call_expr
#     | subscript_expr
class Expression(Subparser):

    def get_prefix_subparser(self, token):
        return self.get_subparser(token, {
            'NUMBER': NumberExpression,
            'STRING': StringExpression,
            'NAME': NameExpression,
            'SELF': SelfReferenceStatement,
            'LPAREN': GroupExpression,
            'LBRACK': ArrayExpression,
            'LBRACE': DictionaryExpression,
            'OPERATOR': UnaryOperatorExpression,
        })

    def get_infix_subparser(self, token):
        return self.get_subparser(token, {
            'OPERATOR': BinaryOperatorExpression,
            'LPAREN': CallExpression,
            'LBRACK': SubscriptOperatorExpression,
        })

    def get_next_precedence(self, tokens):
        if not tokens.is_end():
            token = tokens.current()
            parser = self.get_infix_subparser(token)
            if parser is not None:
                return parser.get_precedence(token)
        return 0

    def parse(self, parser, tokens, precedence=0):
        subparser = self.get_prefix_subparser(tokens.current())

        if subparser is not None:
            left = subparser.parse(parser, tokens)

            if left is not None:
                while precedence < self.get_next_precedence(tokens):
                    op = self.get_infix_subparser(
                        tokens.current()).parse(parser, tokens, left)
                    if op is not None:
                        left = op
                return left


# expr_stmnt: assing_stmnt
#           | expr NEWLINE
class ExpressionStatement(Subparser):

    def parse(self, parser, tokens):
        exp = Expression().parse(parser, tokens)
        if exp is not None:
            if tokens.current().name == 'ASSIGN':
                return AssignmentStatement().parse(parser, tokens, exp)
            elif tokens.current().name == 'OPASSIGN':
                return OperationAssignmentStatement().parse(parser, tokens, exp)
            elif tokens.current().name == 'MEMBER':
                return MemberAccessStatement().parse(parser, tokens, exp)
            else:
                tokens.consume_expected('NEWLINE')
                return exp


# stmnts: stmnt*
class Statements(Subparser):
    def get_statement_subparser(self, token):
        return self.get_subparser(token, {
            'ENTITY': EntityStatement,
            'SELF': SelfReferenceStatement,
            'FUNCTION': FunctionStatement,
            'IF': ConditionalStatement,
            'MATCH': MatchStatement,
            'WHILE': WhileLoopStatement,
            'RETURN': ReturnStatement,
            'EXIT': ExitStatement,
            'NEXT': NextStatement,
        }, ExpressionStatement)

    def parse(self, parser, tokens):
        statements = []
        while not tokens.is_end():
            statement = self.get_statement_subparser(tokens.current()).parse(parser, tokens)
            if statement is not None:
                statements.append(statement)
            else:
                break
        return statements


# prog: stmnts
class Program(Subparser):

    def parse(self, parser, tokens):
        statements = Statements().parse(parser, tokens)
        tokens.expect_end()
        return ast.Program(statements)


# Parser
class Parser(object):

    def __init__(self):
        self.scope = None

    def parse(self, tokens):
        self.scope = []
        return Program().parse(self, tokens)
