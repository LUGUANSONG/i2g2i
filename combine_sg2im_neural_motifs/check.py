import argparse


parser = argparse.ArgumentParser(description='training code')
parser.add_argument("--a", type=int, default=0)
args = parser.parse_known_args()


parser_2 = argparse.ArgumentParser(description='training code')
parser_2.add_argument("--a", type=int, default=0)
parser_2.add_argument("--b", type=int, default=0)
args_2 = parser.parse_args()


print(args)
print(args_2)
