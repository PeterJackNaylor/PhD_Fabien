# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
"""

import pandas as pd
import numpy as np
import os
from optparse import OptionParser
import pdb
import glob

verbose = False
number = 160120
path_file = "C:\Users\Peter\Downloads\id-patient-Sheet1.csv"
path_data = "C:/Data_PHD"
path_input_data = "D:/dataThomas/Projet_FR-TNBC-2015-09-30/All"

def converter(num_str):
    return(num_str[0:3]+num_str[4:7])
    
def convert_data(data, column_name):
    data[column_name] = data.apply (lambda r:  converter(r[column_name]), axis=1).astype(int)

def check_folder(patient_id):
    folder_patient = os.path.join(path_data, patient_id)
    if not os.path.isdir(folder_patient):
        os.mkdir(folder_patient)
        
def getting_files_name(path_input_data):
    return glob.glob(os.path.join(path_input_data,'*.tiff'))
    
def is_match(path, num):
    if verbose:
        if num == number:
            pdb.set_trace()
    try:
        path = path.split('.')[0]
    except:
        pass
    try:
        name = path.split('\\')[-1]
    except:
        name = path
    try:
        lists = name.split('_')
    except:
        lists = [name]
    Matched = False        
    for element in lists:
        try:
            if int(element) == num:
                Matched = True
        except:
            if verbose:
                print 'Not int'
    
    return Matched

def is_match_all(r, folder, id_name = 'InstanceCreationTime'):
    if verbose:
        if r[id_name] == number:
            pdb.set_trace()
    for fn in folder:
        if is_match(fn, r[id_name]):
            return fn
    return '0'   
    
def Matching(data, folder):
    ## If matched: address, else 0
    data["Match"] = data.apply (lambda r:  is_match_all(r, folder), axis=1).astype(str)

def Unmatched_print(data, folder):
    found_association = data.ix[data['Match']!='0'].shape[0]
    to_find = len(folder)
    
    if found_association != to_find:
        print "Their is %d files who don't have a patient name..." %(to_find - found_association)
        matched = np.array(data.ix[data['Match']!='0']["Match"])
        for el in folder:
            if el not in matched:
                print el


if __name__ == "__main__":
    
    data = pd.read_csv(path_file)
    convert_data(data, "patient_id")

    all_tiff = getting_files_name(path_input_data)
    Matching(data, all_tiff)
    
    Unmatched_print(data, all_tiff)
        
    