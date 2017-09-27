
import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors

results = pd.read_csv('results.csv')
grouped_stepsize = results.groupby(['stepsize'])


axes_clearborder = {}
axes_clearborder["Classic"] = 0
axes_clearborder["RemoveBorderWithDWS"] = 2
axes_clearborder["RemoveBorderObjects"] = 1
axes_clearborder["Reconstruction"] = 3

axes_method = {}
axes_method["max"] = 0
axes_method["avg"] = 1
axes_method["median"] = 2
cols = ['Max', 'Average', 'Mediane']
rows = ['Classic', 'BorderObjects', 'BorderObjectsWS', 'Reconstruction']
pad = 5

for name_stepsize, group in grouped_stepsize:
    fig, ax = plt.subplots(ncols=3, nrows=4, sharex=True, sharey=True, figsize=(8., 6.))
    grouped_clearborder_method = group.groupby(['clearborder', 'method'])
    ax[0, 0].set_ylim([0,0.65])
    for name_clearborder_method, group2 in grouped_clearborder_method:
        x_clearborder, y_method = name_clearborder_method
        x, y = axes_clearborder[x_clearborder], axes_method[y_method]
        table = group2.groupby(['lambda']).mean()
        table = table.sort_index()
        AJI = table['AJI']
        lambda_val = [i for i in table.index]
        colors = []
        cmap = cm.get_cmap("gist_earth", len(lambda_val)+1)    # PiYG
        for i in range(cmap.N - 1):
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            colors.append(matplotlib.colors.rgb2hex(rgb))
        width = 1,
        ax[x, y].bar(lambda_val, list(AJI), color=colors, alpha=1., width=width)
        ax[x, y].axhline(y=np.max(AJI),xmin=0,xmax=3,c="black",ls="--",linewidth=0.5,zorder=0)
        ax[x, y].text(7.5, np.max(AJI) + 0.01, "{0:.3}".format(np.max(AJI)), rotation=0)
    for a, col in zip(ax[0], cols): 
        a.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    ha='center', va='baseline')

    for a, row in zip(ax[:,0], rows):
        a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                    xycoords=a.yaxis.label, textcoords='offset points',
                    ha='right', va='center', rotation="vertical")
    ax[0,-1].set_xticks(ticks=lambda_val)
    ax[-2,1].set_xticks(ticks=lambda_val)
    ax[-2,2].set_xticks(ticks=lambda_val)
    ax[0,0].set_ylabel('AJI')
    ax[1,0].set_ylabel('AJI')
    ax[2,0].set_ylabel('AJI')
    ax[3,0].set_ylabel('AJI')
    ax[-1,0].set_xlabel('lambda')
    ax[-2,1].set_xlabel('lambda')
    ax[-2,2].set_xlabel('lambda')
    ax[-1,1].axis('off')
    ax[-1,2].axis('off')
    fig.savefig(str(name_stepsize) + ".png", bbox_inches='tight')


results = results[results["method"] == "max"]
results = results[results["clearborder"] == "RemoveBorderWithDWS"]
grouped_stepsize = results.groupby(['stepsize'])
AJI_stepsize = []
time_stepsize = []
stepsize = []
n_img = []
for name_stepsize, group in grouped_stepsize:
    best_AJI = group['AJI'].max()  
    AJI_stepsize.append(best_AJI)
    n_img.append(group['n_img'].mean())
    time = group['time'].mean() * group['n_img'].mean()
    time_stepsize.append(time)
    stepsize.append(name_stepsize)

fig, axes = plt.subplots(ncols=3, figsize=(10.,4.))
colors = []
cmap = cm.get_cmap("ocean", len(stepsize) + 1)    # PiYG
for i in range(cmap.N - 1):
    rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    colors.append(matplotlib.colors.rgb2hex(rgb))

axes[0].bar(range(len(stepsize)), AJI_stepsize, color=colors, alpha=1., width=width)
axes[0].xaxis.set(ticks=range(len(stepsize)), ticklabels=stepsize)
axes[0].set_xlabel('Stepsize', rotation=45)
axes[0].set_ylabel('AJI', rotation=45)
axes[0].xaxis.set_label_coords(1.0, -0.025)
axes[0].yaxis.set_label_coords(-0.025, 1.)        
axes[0].axhline(y=np.max(AJI_stepsize),xmin=0,xmax=3,c="black",ls="--",linewidth=0.5,zorder=0)
axes[0].text(2.5, np.max(AJI_stepsize) + 0.01, "{0:.3}".format(np.max(AJI_stepsize)), rotation=0)
axes[0].set_ylim([0,0.55])

axes[1].plot(stepsize, time_stepsize, '-')
axes[1].set_xlabel('Stepsize', rotation = 45)
axes[1].set_ylabel('Time (s)', rotation = 45)
axes[1].xaxis.set_label_coords(1.0, -0.025)
axes[1].yaxis.set_label_coords(-0.025, 1.0)

axes[2].set_xlabel('N image', rotation = 45)
axes[2].set_ylabel('Time (s)', rotation = 45)
axes[2].xaxis.set_label_coords(1.1, 0.0)
axes[2].yaxis.set_label_coords(-0.025, 1.0)
axes[2].plot(n_img, time_stepsize, '-')

fig.text(0.05, 1.05, 'Method: Max', rotation=0)
fig.text(0.05, 1., 'Border: RemoveBorderWithDWS', rotation=0)

for tick in axes[0].get_xticklabels():
    tick.set_rotation(45)
for tick in axes[0].get_yticklabels():
    tick.set_rotation(45)
for tick in axes[1].get_xticklabels():
    tick.set_rotation(45)
for tick in axes[1].get_yticklabels():
    tick.set_rotation(45)
for tick in axes[2].get_xticklabels():
    tick.set_rotation(45)
for tick in axes[2].get_yticklabels():
    tick.set_rotation(45)
plt.savefig("StepsizeEffect.png", bbox_inches='tight')