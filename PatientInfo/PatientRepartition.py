import pandas as pd
import os


def check_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def ImportData(name):
    data = pd.read_excel(name, sheetname=1)
    data = data[pd.notnull(data["RCB"])]
    return data

import pdb


def converter(num_str):
    if num_str == "hors Curie":
        return 0
    num_str = str(num_str)
    if len(num_str) == 7:
        return(num_str[0:3] + num_str[4:7])
    else:
        return num_str


def converter2(num_str):
    return(num_str[0:3] + num_str[4:7])


def CheckAndWrite(path, field1, field2):
    if field1 == "":
        field1 = path
        return field1, field2
    elif field2 == "":
        field2 = path
        return field1, field2
    else:
        print "their is a problem..."
        pdb.set_trace()


def MatchMaybe4(name, Biopsie, Piece_operatoire, data_piece_file, files):
    fold_bio = ""
    fold_bio2 = ""
    fold_pie = ""
    fold_pie2 = ""
    Biopsie = str(Biopsie)
    Piece_operatoire = str(Piece_operatoire)
    for f in files:
        try:
            if "1572_HES_" not in f:
                if Biopsie in f:
                    fold_bio, fold_bio2 = CheckAndWrite(f, fold_bio, fold_bio2)
                if Piece_operatoire != "0" and Piece_operatoire in f:
                    fold_pie, fold_pie2 = CheckAndWrite(f, fold_pie, fold_pie2)
            else:
                small_data = data_piece_file[data_piece_file["Match"] == f]

                pat_id = int(small_data["patient_id"])
                if int(Biopsie) == pat_id:
                    fold_bio, fold_bio2 = CheckAndWrite(f, fold_bio, fold_bio2)
                elif int(Piece_operatoire) == pat_id:
                    fold_pie, fold_pie2 = CheckAndWrite(f, fold_pie, fold_pie2)
        except:
            print 'PLEASE CHECK'
            pdb.set_trace()
    return pd.Series({"Biopsie_file": fold_bio, "Biopsie_file2": fold_bio2,
                      "Piece_operatoire_file": fold_pie, "Piece_operatoire_file2": fold_pie2})


def where_is_file(data_patient_piece, data_piece_file, files):
    data_patient_piece[["Biopsie_file", "Biopsie_file2",
                        "Piece_operatoire_file", "Piece_operatoire_file2"]] = data_patient_piece.apply(
        lambda r: MatchMaybe4(r.name, r["Biopsie"], r["Piece_operatoire"],
                              data_piece_file, files), axis=1)


def convert_data(data, column_name):
    data[column_name] = data.apply(
        lambda r:  converter(r[column_name]), axis=1).astype(int)


def convert2_data(data, column_name):
    data[column_name] = data.apply(
        lambda r:  converter2(r[column_name]), axis=1).astype(int)


def IdBioWho(data):
    return data[["Dossier", "Biopsie", "Piece_operatoire"]]


def findAllFiles(path, File_extension):
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(File_extension):
                res.append(os.path.join(root, file))
    return res


def MoveAndRename(r, path):
    dossier = r["Dossier"]
    biopsie = r["Biopsie"]
    Piece_operatoire = r["Piece_operatoire"]
    bio_file = r["Biopsie_file"]
    bio_file2 = r["Biopsie_file2"]
    pie_file = r["Piece_operatoire_file"]
    pie_file2 = r["Piece_operatoire_file2"]

    root_folder = os.path.join(path, str(dossier))

    i = 0

    if bio_file == "" and pie_file == "" and bio_file2 == "" and pie_file2 == "":
        print "Nothing for {}".format(dossier)

    if bio_file != "":
        i += 1
        check_folder(root_folder)
        new_filename = os.path.join(root_folder, str(biopsie) + "_biopsy.tiff")
        print ">>>>>>> {} --------> {}".format(bio_file, new_filename)

    if bio_file2 != "":
        i += 1
        check_folder(root_folder)
        new_filename = os.path.join(
            root_folder, str(biopsie) + "_secondbiopsy.tiff")
        print ">>>>>>> {} --------> {}".format(bio_file2, new_filename)
    j = 0
    if pie_file != "":
        j += 1
        check_folder(root_folder)
        new_filename = os.path.join(
            root_folder, str(Piece_operatoire) + "_piece.tiff")
        print ">>>>>>> {} --------> {}".format(pie_file, new_filename)

    if pie_file2 != "":
        j += 1
        check_folder(root_folder)
        new_filename = os.path.join(
            root_folder, str(biopsie) + "_secondpiece.tiff")
        print ">>>>>>> {} --------> {}".format(bio_file, new_filename)

    print "{} biopsie files and {} Pieces for dossier {}".format(i, j, dossier)
    if i + j == 1:
        print ">>>>>>>>>>>>>>>>>> LOOK HERE <<<<<<<<<<<<<<<<<<<<"
    return pd.Series({"Biopsy_here": min(1, i), "Piece_here": min(1, j)})

def GetOptions(verbose=True):

    parser = OptionParser()
    parser.add_option('--pi', dest="patient_info", type="string",
                      help="patient info xlsx file")
    parser.add_option("--f", dest="folder", type='string', 
                     default="/media/naylor/F00E67D40E679300/Projet_FR-TNBC-2015-09-30/All",
                     help="folder where the histopathology files are")
    (options, args) = parser.parse_args()

    if verbose:
        print " \n "
        print "Input paramters to run:"
        print " \n "
        print "Patient info file     : | {}".format(options.patient_info)
        print "Biopsy and WSI folder : | {}".format(options.folder)


    return (options, args)


if __name__ == "__main__":

    options, args = GetOptions()
    name = options.patient_info
    data_patient_piece = ImportData(name)
    data_patient_piece = IdBioWho(data_patient_piece)

    convert_data(data_patient_piece, "Biopsie")
    convert_data(data_patient_piece, "Piece_operatoire")

    path_file = "id-patient-Sheet1.csv"
    path_input_data = options.folder

    data_piece_file = pd.read_csv(path_file)
    convert2_data(data_piece_file, "patient_id")
    all_tiff = getting_files_name(path_input_data)
    Matching(data_piece_file, all_tiff)
    data_piece_file = data_piece_file[data_piece_file["Match"] != "0"]

    files = findAllFiles(path_input_data, ".tiff")

    where_is_file(data_patient_piece, data_piece_file, files)

    data_patient_piece[["Biopsy_here", "Piece_here"]] = data_patient_piece.apply(
        lambda r: MoveAndRename(r, path_input_data), axis=1)
