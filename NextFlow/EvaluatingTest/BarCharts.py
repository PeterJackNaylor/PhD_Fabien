import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches


table = pd.read_csv("results.csv")
table.loc[table["Model"] == "DeconvNet_0.01_0.99_0.0005", "Model"] = "DeconvNet"
table.loc[table["Model"] == "FCN_0.01_0.99_0.005", "Model"] = "FCN"
table.loc[table["Model"] == "ensemble", "Model"] = "Ensemble (FCN + DeconvNet)"
table.loc[table["Model"] == "classic", "Model"] = "UNet"
table.loc[table["Model"] == "no_elast", "Model"] = "UNet without elast aug"
table.loc[table["Model"] == "no_he", "Model"] = "UNet without HE aug"
table.loc[table["Model"] == "no_hsv", "Model"] = "UNet without HSV aug"
table.loc[table["Model"] == "no_hsv_he", "Model"] = "UNet without HSV/HE aug"
table.loc[table["Model"] == "no_hsv_he_elast", "Model"] = "UNet without HSV/HE/elast aug"
table.loc[table["Model"] == "nothing", "Model"] = "UNet without any augmentation"



grouped = table.groupby(['Model'])
fig, ax = plt.subplots(nrows=5, sharey=True, figsize=(30., 10.0))
colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
          "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]

Patches = []

for i, (name, group) in enumerate(grouped):
    group = group.sort_values(['Image'])
    ACC = group["ACC"]
    F1 = group["F1"]
    AJI = group["AJI"]
    Recall = group["Recall"]
    Precision = group["Precision"]

    ind = np.arange(len(group.index))
    width = 0.08
    rectsACC = ax[0].bar(ind + i * width, ACC, width, color=colors[i])
    rectsF1 = ax[1].bar(ind + i * width, F1, width, color=colors[i])
    rectsAJI = ax[2].bar(ind + i * width, AJI, width, color=colors[i])
    rectsPrec = ax[3].bar(ind + i * width, Precision, width, color=colors[i])
    rectsReca = ax[4].bar(ind + i * width, Recall, width, color=colors[i])
    patch_to_add = mpatches.Patch(color=colors[i], label=name)
    Patches.append(patch_to_add)

ax[0].set_ylabel('ACC')
ax[1].set_ylabel('F1')
ax[2].set_ylabel('AJI')
ax[3].set_ylabel('Precision')
ax[4].set_ylabel('Recall')

Image_names = [el.replace("_", " ") for el in group['Image']]
#Image_names = ["offset"] + Image_names
for j in range(5):
    ax[j].set_xticks(ind + 9 * width / 2)
    ax[j].set_xticklabels(Image_names)
    ax[j].axhline(y=0.75,xmin=0,xmax=3,c="gray",ls="--",linewidth=0.5,zorder=0)
    ax[j].axhline(y=0.50,xmin=0,xmax=3,c="black",ls="-.",linewidth=0.5,zorder=0)
    ax[j].axhline(y=0.25,xmin=0,xmax=3,c="gray",ls="--",linewidth=0.5,zorder=0)

box = ax[4].get_position()
ax[4].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax[4].legend(handles=Patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)


#set_matplotlib_formats('pdf', 'svg')
fig.savefig("BarPlotResult.png")
