import os, glob
import pandas as pd
import re
import time

#table=the csv containing the joined parcel and point data fromt he original test sample

#simple exporitory functions to see how many duplicate records there are in the dataset
unq=table.OWN1.unique()
p1_count=len(table["OWN1"])-len(unq)


#the initial creation of the honorific list and name abbreviation dictionary


NameCleaner= ["DR.","MR.", "MS.", "MRS.", "CAPTAIN", "CPT.", "PROF", "REV." "COACH", "PROFESSOR", "REVEREND" ,"SIR", "LT.", "SGT."]


#note, i grabbed the dictionary from a wikioedia page and created a csv from that. 
abbr=pd.read_csv("abbr.csv", low_memory=False, names=["Ab", "Full"])

NamesExpander={"Chr.": "Christophe"}
for i in range(len(abbr)):
    NamesExpander.update( {abbr["Ab"][i] : abbr["Full"][i]} )
    
#The actuall function that access each record and passes them through the drop list and expander dictionary. 
#Note: i am only analyzing the first 10000 records here for expediancies' sake

def __calculate_name_matching(row):
    for i in range(10000):
        y= str(row["OWN1"][i]).upper()
        #print(1,y)
        for clean in NameCleaner:
            y=y.replace(clean ,'')            
        for expand in NamesExpander:
            y = y.replace(expand, NamesExpander[expand])
        row["OWN1"][i]=y
        #print(2,y)
#To remove single letters
#y =  re.sub(r"\b[a-zA-Z]\b", "", y)


#This is simply a means to calculate runtime
t1=time.time()
__calculate_name_matching(table)
t2=time.time()


#This simply an effort to see how many changes were made. 
#Note the initial change volume for 10000 records is 2 changes

unq2=table.OWN1.unique()
p2_count=len(table["OWN1"])-len(unq2)

print(p1_count, p2_count)
