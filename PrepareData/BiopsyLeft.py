import pandas as pd
import os 
from glob import glob
from os.path import basename

excel_name = 'MarickDataSet.xlsx'
sheet = 'TRIPLE NEGATIF'
table = pd.read_excel(excel_name, sheet)

table = table[pd.notnull(table.RCB)]
table = table[['Dossier', 'Biopsy', 'RCB']]
table['Present'] = 0
PATH_TIFF = glob('/media/pnaylor/Peter-Work/Projet_FR-TNBC-2015-09-30/All/Biopsy/*.tiff')
for f in PATH_TIFF:
    name = basename(f).split('.')[0]
    tmp_table = table.copy()
    tmp_table = tmp_table[tmp_table["Biopsy"] == int(name)]
    ind = tmp_table.index[0]
    try:
        table.iloc[ind, 'Present'] = 1
    except:
        table.ix[ind, 'Present'] = 1
final_tab = table[table['Present'] == 0]
final_tab = final_tab.drop('Present', axis=1)
final_tab.to_excel('MissingData.xlsx', sheet_name='Missing Biopsy')