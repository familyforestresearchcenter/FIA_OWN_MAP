from metaphone import doublemetaphone
from enum import Enum

class Threshold(Enum):
    WEAK = 0
    NORMAL = 1
    STRONG = 2
    
def double_metaphone(value):
    print(doublemetaphone(value))
    return doublemetaphone(value)


#(Primary Key = Primary Key) = Strongest Match
#(Secondary Key = Primary Key) = Normal Match
#(Primary Key = Secondary Key) = Normal Match
#(Alternate Key = Alternate Key) = Minimal Match
def double_metaphone_compare(tuple1,tuple2,threshold):
    if threshold == Threshold.WEAK:
        if tuple1[1] == tuple2[1]:
            return True
    elif threshold == Threshold.NORMAL:
        if tuple1[0] == tuple2[1] or tuple1[1] == tuple2[0]:
            return True
    else:
        if tuple1[0] == tuple2[0]:
            return True
    return False
    
 
 
 what follows is my own testing of the functions

tuple1=double_metaphone("Vance Allen Harris Jr")
tuple2=double_metaphone("Jesse Caputo")
tuple3=double_metaphone("Brett Butler")

double_metaphone_compare(tuple1, tuple2, Threshold.STRONG)
