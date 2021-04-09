"""
CSC413 Final Project, W2021
University of Toronto

By Henry Tu & Seel Patel

#################################

Structural similarity Analysis
"""

import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 %s <path to dataset pickle file>' % sys.argv[0])
        exit(1)