import pandas as pd
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdb
from matplotlib import lines
CSV = glob('*.csv')

df_s = []
for f in CSV:
    tmp = pd.read_csv(f, index_col=0)
    if "FCN" == f.split('_')[0]:
        model, organ = f.split("__")[0].split('_')
        id_model = f.split("__")[1].split('.')[0]
        tmp["ORGAN"] = organ
    else:
        model = f.split('__')[1].split('.')[0]
        id_model = f.split("__")[0]

    tmp["MODEL"] = model
    tmp["Model_Unique"] = id_model
    df_s.append(tmp)

table = pd.concat(df_s)
table.to_csv('Test_HP.csv')


new_table = table.groupby(['Model_Unique']).mean()
idx = new_table.groupby(['MODEL'])['AJI'].transform(max) == new_table['AJI']
winners = np.unique(new_table[idx]['Model_Unique'])

table = table.loc[table.Model_Unique.isin(winners)]

grouped = table.groupby(['ORGAN', 'NUMBER'])

fig, ax = plt.subplots(nrows=3, sharey=True, figsize=(15.0, 10.0))
colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
          "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]

Patches = []
methods = np.unique(table["Model"])
organ_order = np.unique(table["ORGAN"])

mod = {el:[0.] * len(organ_order) * len(np.unique(table["NUMBER"])) for el in methods}
mod2 = {el:[0.] * len(organ_order) * len(np.unique(table["NUMBER"])) for el in methods}
mod3 = {el:[0.] * len(organ_order) * len(np.unique(table["NUMBER"])) for el in methods}
dic = {"ACC":mod, "F1":mod2, "AJI":mod3}

def fill_dic(row, ind):
    acc = row["ACC"]
    f1 = row["F1"]
    aji = row["AJI"]
    model = row["Model"]
    dic["ACC"][model][ind] = acc
    dic["F1"][model][ind] = f1
    dic["AJI"][model][ind] = aji

for i, ((slide, num), tble) in enumerate(grouped):
    ind = 2 * np.where(organ_order == slide)[0][0] + int(num)
    tble.apply(lambda row: fill_dic(row, ind), axis=1)

width = 0.2
for j, model in enumerate(methods):
    ind = np.arange(len(dic["ACC"][model]))
    rectsACC = ax[0].bar(ind + j * width, dic["ACC"][model], width, color=colors[j])
    rectsAJI = ax[1].bar(ind + j * width, dic["F1"][model], width, color=colors[j])
    rectsF1 = ax[2].bar(ind + j * width, dic["AJI"][model], width, color=colors[j])
    patch_to_add = mpatches.Patch(color=colors[j], label=model)
    Patches.append(patch_to_add)

ax[0].set_ylabel('ACC')
ax[1].set_ylabel('F1')
ax[2].set_ylabel('AJI')

Image_names = []
for el in organ_order:
    Image_names.append("1")
    Image_names.append("2")

for j in range(3):
    if j == 2:
        ax[j].set_xticks(ind + 3 * width / 2)
        ax[j].set_xticklabels(Image_names, rotation=0)
    else:
        ax[j].set_xticks(ind + 3 * width / 2)
        ax[j].set_xticklabels(Image_names, rotation=0)
    ax[j].axhline(y=0.75,xmin=0,xmax=3,c="gray",ls="--",linewidth=0.5,zorder=0)
    ax[j].axhline(y=0.50,xmin=0,xmax=3,c="black",ls="-.",linewidth=0.5,zorder=0)
    ax[j].axhline(y=0.25,xmin=0,xmax=3,c="gray",ls="--",linewidth=0.5,zorder=0)
box = ax[2].get_position()
ax[2].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax[2].legend(handles=Patches, loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=5)

ax2 = plt.axes([0, 0, 1, 1])
ax2.set_position([box.x0 + 0.05, box.y0 + box.height * 0.1 - 0.15,
                 box.width - 0.08, box.height * 0.9])
ax2.axis('off')
for j, organ in enumerate(organ_order):
    P0 = [ind[j] * 2 -2, -2]
    P1 = [ind[j] * 2 + 1 -2, -2]
    line_x, line_y = np.array([P0, P1])
    line_x, line_y = np.array([[0.05, 0.05], [0.05, 0.55]])
    line1 = lines.Line2D(line_x, line_y, lw=2., color='k')
    #ax2.add_line(line1)
    alpha = 0.861
    xmin = (float(ind[j * 2]) / np.max(ind) + 0.01 + 0.16) * alpha + 0.006
    xmax = (float(ind[j * 2 + 1]) / np.max(ind) - 0.01 )  * alpha
    XY = np.mean([xmin, xmax])
    fs = 6.
    ax[2].annotate(organ, xy=(XY, -0.15), xytext=(XY, -0.2), xycoords='axes fraction',
                fontsize=fs*1.5, ha='center', va='top',
                bbox=dict(boxstyle='square', fc='white'),
                arrowprops=dict(arrowstyle='-[, widthB=5.65, lengthB=0.5', lw=2.0))

# #set_matplotlib_formats('pdf', 'svg')
fig.savefig("VSPaper.png",bbox_inches='tight')

