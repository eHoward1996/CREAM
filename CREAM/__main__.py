"""
Main
----

Command line interface.
"""
import argparse
import CREAM.interpreter as interpreter


version = '0.0.1'

try:
    input = raw_input
except NameError:
    pass


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', '--trace', action='store_true')
    argparser.add_argument('file', nargs='?')
    return argparser.parse_args()


def interpret_file(path, trace=False):
    if path[-4:] != '.crm':
        raise Exception('Invalid file type! Files must end with ".crm"')
    with open(path) as f:
        prgrm = interpreter.evaluate(f.read(), trace=trace)
        if prgrm is not None:
            print(prgrm)


def repl():
    print('CREAM {}. Press Ctrl+C to exit.'.format(version))
    env = interpreter.create_global_env()
    buf = ''
    try:
        while True:
            inp = input('>>> ' if not buf else '')
            if inp == '':
                print(interpreter.evaluate_env(buf, env))
                buf = ''
            else:
                buf += '\n' + inp
    except KeyboardInterrupt:
        pass


def main():
    args = parse_args()
    if args.file:
        interpret_file(args.file, args.trace)
    else:
        repl()


if __name__ == '__main__':
    main()
