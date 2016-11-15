# -*- coding: utf-8 -*-
"""
Created on Tue May 03 16:32:53 2016

@author: Peter
"""

import pandas as pd
import numpy as np
import os
import pdb
import glob

verbose = False
number = 160120
path_file = "id-patient-Sheet1.csv"
path_input_data = "/media/naylor/F00E67D40E679300/Projet_FR-TNBC-2015-09-30/All"
rename = True


def converter(num_str):
    return(num_str[0:3] + num_str[4:7])


def convert_data(data, column_name):
    data[column_name] = data.apply(
        lambda r:  converter(r[column_name]), axis=1).astype(int)


def check_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def getting_files_name(path_input_data):
    return glob.glob(os.path.join(path_input_data, '*.tiff'))


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


def is_match_all(r, folder, id_name='InstanceCreationTime'):
    if verbose:
        if r[id_name] == number:
            pdb.set_trace()
    for fn in folder:
        if is_match(fn, r[id_name]):
            return fn
    return '0'


def Matching(data, folder):
    # If matched: address, else 0
    data["Match"] = data.apply(
        lambda r:  is_match_all(r, folder), axis=1).astype(str)


def Unmatched_print(data, folder):
    found_association = data.ix[data['Match'] != '0'].shape[0]
    to_find = len(folder)

    if found_association != to_find:
        print "Their is %d files who don't have a patient name..." % (to_find - found_association)
        matched = np.array(data.ix[data['Match'] != '0']["Match"])
        for el in folder:
            if el not in matched:
                print el


def MissingFiles(data, column_match="Match", id_name="patient_id", path=path_input_data):
    temp = data.ix[data[column_match] == '0'][id_name]
    i = 0
    Biop = os.path.join(path, "Biopsy")
    WTS = os.path.join(path, "WholeTumorSlide")
    for row in temp.index:
        # print "Slide %d missing, id number %d" %(row+1, data.ix[row,id_name])
        id_ = data.ix[row, id_name]
        if id_ == 0:
            pdb.set_trace()
        fileWTS = os.path.join(WTS, str(id_) + ".tiff")
        fileBiop = os.path.join(Biop, str(id_) + ".tiff")
        if not (os.path.isfile(fileWTS) or os.path.isfile(fileBiop)):
            i += 1
            print "Slide %d, id number %d" % (row + 1, id_)
    print "Their is %d missing slides" % i


def RenamingAndMoving(data, path, ColumnBiopsy="Biopsy", id_name="patient_id", column_address="Match"):
    path_biopsy = os.path.join(path, "Biopsy")
    path_full_tumor = os.path.join(path, "WholeTumorSlide")

    check_folder(path_biopsy)
    check_folder(path_full_tumor)

    #data = data[data[column_address] != "0"]

    DataBiopsy = data[data[ColumnBiopsy] == 1]
    DataWTS = data[data[ColumnBiopsy] == 0]

    for el in DataBiopsy.index:
        try:
            original_file = DataBiopsy.ix[el, column_address]
            new_file = os.path.join(path, "Biopsy", str(
                DataBiopsy.ix[el, id_name]) + ".tiff")
            if '537757' in new_file:
                pdb.set_trace()
            os.rename(original_file, new_file)
        except:
            if not os.path.isfile(new_file):
                print "Can't change %d file" % DataBiopsy.ix[el, id_name]
    for el in DataWTS.index:
        try:
            original_file = DataWTS.ix[el, column_address]
            new_file = os.path.join(path, "WholeTumorSlide", str(
                DataWTS.ix[el, id_name]) + ".tiff")
            os.rename(original_file, new_file)
        except:
            if not os.path.isfile(new_file):
                print "Can't change %d file" % DataWTS.ix[el, id_name]

if __name__ == "__main__":

    data = pd.read_csv(path_file)
    convert_data(data, "patient_id")

    all_tiff = getting_files_name(path_input_data)
    Matching(data, all_tiff)

    Unmatched_print(data, all_tiff)
    MissingFiles(data)
