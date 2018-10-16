#!/usr/bin/env python

import argparse
import json
from pprint import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, metavar='FILE')
    parser.add_argument('-d', '--depth', type=int, default=4, metavar='N')

    args = parser.parse_args()
    with open(args.input, 'r') as f:
        pprint(json.load(f), depth=args.depth)

if __name__ == '__main__':
    main()