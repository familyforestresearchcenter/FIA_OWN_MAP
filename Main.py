#!/usr/bin/env python
# coding: utf-8

import os
import sys
from zipfile import ZipFile

sys.path.insert(0, os.path.abspath('.'))

from utils.helpers import *
from Name_Matching import *
from Classify_Unknowns import *
from Land_Analysis import *
# from Summary_Script import *
import shutil

if __name__ == '__main__':
    
    t1 = time.time()
    
    exec(open("Preprocessing.py").read())
    exec(open("Classify_Unknowns.py").read())
    exec(open("Name_Matching_2_8.py").read())
    os.remove(f'./data/new_parallel_state_temp.csv')
    exec(open("Summary_Script.py").read())
    os.remove(f'./data/Full_Data_Table.csv')
    exec(open("New_Map_Data.py").read())
    state_name = [i for i in os.listdir('./data') if len(i) == 2][0]
    shutil.make_archive(f'./data/{state_name}', 'zip', f'./data/{state_name}')
    shutil.rmtree(f'./data/{state_name}')
    
    
    
    t2 = time.time()
    print(f'{state_name} Completed in : {t2-t1} s\n')
