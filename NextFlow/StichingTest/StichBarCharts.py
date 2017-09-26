
import matplotlib
matplotlib.use('agg')
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors

results = pd.read_csv('results.csv')
grouped_stepsize = results.groupby(['stepsize'])


axes_clearborder = {}
axes_clearborder["Classic"] = 0
axes_clearborder["RemoveBorderWithDWS"] = 1
axes_clearborder["RemoveBorderObjects"] = 2
axes_clearborder["Reconstruction"] = 3

axes_method = {}
axes_method["max"] = 0
axes_method["avg"] = 1
axes_method["median"] = 2

for name_stepsize, group in grouped_stepsize:
    fig, ax = plt.subplots(ncols=3, nrows=4, sharex=True, sharey=True, figsize=(8., 6.))
    grouped_clearborder_method = group.groupby(['clearborder', 'method'])

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
    ax[0,-1].set_xticks(ticks=lambda_val)
    ax[-1,-1].set_xlabel('lambda')
    fig.savefig(str(name_stepsize) + ".png", bbox_inches='tight')


results = results[results["method"] == "max"]
results = results[results["clearborder"] == "RemoveBorderWithDWS"]
grouped_stepsize = results.groupby(['stepsize'])
fig, axes = plt.subplots(ncols=2, figsize=(8.,4.))
AJI_stepsize = []
time_stepsize = []
stepsize = []
for name_stepsize, group in grouped_stepsize:
    best_AJI = group['AJI'].max()  
    AJI_stepsize.append(best_AJI)

    time = group['time'].mean() * group['n_img'].mean()
    time_stepsize.append(time)
    stepsize.append(name_stepsize)

colors = []
cmap = cm.get_cmap("ocean", len(stepsize) + 1)    # PiYG
for i in range(cmap.N - 1):
    rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    colors.append(matplotlib.colors.rgb2hex(rgb))
axes[0].bar(range(len(stepsize)), AJI_stepsize, color=colors, alpha=1., width=width)
axes[0].xaxis.set(ticks=range(len(stepsize)), ticklabels=stepsize)
axes[1].plot(stepsize, time_stepsize, '-')
for tick in axes[0].get_xticklabels():
    tick.set_rotation(45)
plt.savefig("StepsizeEffect.png")