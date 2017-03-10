import glob as g
import os
import random
files_be = g.glob('/home/naylor/Bureau/BaochuanPang/GT_111111/*benign*.tif')
files_ma = g.glob('/home/naylor/Bureau/BaochuanPang/GT_111111/*malignant*.tif')


for i in range(26):
    patient = "Slide_{}".format(i)
    patient_gt = patient.replace("Slide", "GT")

    os.mkdir('/home/naylor/Bureau/BaochuanPang/' + patient)
    os.mkdir('/home/naylor/Bureau/BaochuanPang/' + patient_gt)

    i_rand = random.randint(0, len(files_be) - 1)

    benign_gt = files_be.pop(i_rand)
    benign_sl = benign_gt.replace("GT", "Slide")
    os.rename(benign_gt, benign_gt.replace('GT_111111', patient_gt))
    os.rename(benign_sl, benign_sl.replace('Slide_111111', patient))
    if i < 6:
        i_rand = random.randint(0, len(files_be) - 1)

        benign_gt = files_be.pop(i_rand)
        benign_sl = benign_gt.replace("GT", "Slide")
        os.rename(benign_gt, benign_gt.replace('GT_111111', patient_gt))
        os.rename(benign_sl, benign_sl.replace('Slide_111111', patient))

    i_rand = random.randint(0, len(files_ma) - 1)

    malignant_gt = files_ma.pop(i_rand)
    malignant_sl = malignant_gt.replace("GT", "Slide")
    os.rename(malignant_gt, malignant_gt.replace('GT_111111', patient_gt))
    os.rename(malignant_sl, malignant_sl.replace('Slide_111111', patient))
