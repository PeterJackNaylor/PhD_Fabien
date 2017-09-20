from glob import glob 
import os



def ChangeDirrectory(foldername, old_id, new_id):
    os.rename(foldername, foldername.replace(old_id, new_id))
    files_to_change = glob(foldername.replace(old_id, new_id) + "/*.png")
    for subfile in files_to_change:
        os.rename(subfile, subfile.replace("/" + old_id + "_", "/" + new_id + "_")) 

Mapping = {}
Mapping["141549"] = "02"
Mapping["160120"] = "07"
Mapping["162438"] = "06"
Mapping["498959"] = "04"
Mapping["508389"] = "08"
Mapping["536266"] = "09"
Mapping["544161"] = "10"
Mapping["572123"] = "03"
Mapping["574527"] = "11"
Mapping["581910"] = "01"
Mapping["588626"] = "05"


for id_key in Mapping.keys():
    val = Mapping[id_key]
    ChangeDirrectory('./ToAnnotateColor/Slide_{}'.format(val), val, id_key)
    ChangeDirrectory('./ToAnnotateColor/GT_{}'.format(val), val, id_key)

