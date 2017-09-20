from glob import glob 
import os



def ChangeDirrectory(foldername, old_id, new_id):
    for subfile in glob(foldername + "/*.png"):
        os.rename(subfile, subfile.replace(old_id, new_id))
    os.rename(foldername, foldername.replace(old_id, new_id))


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

