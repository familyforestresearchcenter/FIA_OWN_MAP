#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from utils.helpers import *
from Name_Matching import *
from Classify_Unknowns import *
from Land_Analysis import *
from Summary_Script import *

if __name__ == '__main__':

    #state_name = 'FAKE'

    t1 = time.time()

    exec(open("Name_Matching.py").read())
    exec(open("Classify_Unknowns.py").read())
    exec(open("Land_Analysis.py").read())
    exec(open("Summary_Script.py").read())

    t2 = time.time()
    print(f'completed in : {t2-t1} s\n')
