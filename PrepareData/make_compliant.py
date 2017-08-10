import os
import shutil

IN_DIR = '/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/masks'
OUT_DIR = '/Users/naylorpeter/Documents/Histopathologie/CellCognition/export/masks_renamed'
plates = filter(lambda x: os.path.isdir(
    os.path.join(IN_DIR, x)), os.listdir(IN_DIR))

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

for plate in plates:
    if not os.path.isdir(os.path.join(OUT_DIR, plate)):
        os.makedirs(os.path.join(OUT_DIR, plate))

    print plate
    images = filter(lambda x: os.path.splitext(
        x)[-1].lower() == '.png', os.listdir(os.path.join(IN_DIR, plate)))
    for img in images:
        #^%(plate)s$/^%(pos)s$/.*P%(pos)s_T%(time)05d_C%(channel)s_Z%(zslice)d_S1.tif
        #^%(plate)s$/Mask_%(pos)s__T%(time)02d__%(channel)s__Z%(zslice)02d.png

        base_name = os.path.splitext(img)[0]
        out_name = '%s__T00__c00__Z00.png' % base_name
        print base_name, out_name
        #shutil.copy(os.path.join(IN_DIR, plate, img),
        #            os.path.join(OUT_DIR, plate, out_name))
