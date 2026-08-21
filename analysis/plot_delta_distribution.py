# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %%

with open("dataset/main.json", "r") as f:
    d = json.load(f)
# %%
a = np.zeros(1000)
for item in d:
    id = item['scenario_id']
    val = 0
    num = 0
    for b in item['benchmarks']:
        val += - b['incumbent_score'] + b['challenger_score']
        num += 1
    
    a[id] = val / num
    
a

darkblue = '#3d348b'
blue = "#7678ed"
yellow = '#f7b801'
orange = '#f18701'
darkorange = '#f35b04'

# %%
fontcolor = '#111'
sns.set_style('white')
plt.figure(figsize=(10, 4))

plt.rcParams['axes.labelcolor'] = fontcolor
plt.rcParams['xtick.color'] = fontcolor
plt.rcParams['ytick.color'] = fontcolor
plt.rcParams['text.color'] = fontcolor
plt.rcParams['axes.edgecolor'] = fontcolor
plt.rcParams['axes.titlecolor'] = fontcolor
plt.rcParams['font.size'] = 20

# ---- Histogram (NO KDE here) ----
ax = sns.histplot(
    a,
    bins=14,
    stat='percent',
    color=darkorange,
    edgecolor='white',
    linewidth=6
)

# ---- Reflection KDE ----
a_reflect = np.concatenate([a, -a])


sns.kdeplot(
    a_reflect,
    clip=(0, 5),
    linewidth=5,
    linestyle='-',
    alpha=0.7,
    color=yellow,
    ax=ax,
    common_norm=False
)

# rescale KDE from density → percent
line = ax.lines[-1]
x, y = line.get_data()
line.set_data(x, y * 75)

# ---- Styling ----
ax.spines["top"].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)

plt.xlabel("∆", fontsize=30, fontfamily='serif')
plt.ylabel("", fontsize=1)

plt.xticks(fontsize=20)
plt.yticks([5, 10], ['5%', '10%'], fontsize=20)

plt.xlim(0, 5)
plt.grid(True, axis='y', linestyle='-', alpha=1)

plt.axvline(
    a.mean(),
    color=blue,
    linestyle='--',
    linewidth=5,
    label=f'Mean: {a.mean():.2f}'
)

plt.legend(fontsize=20)
plt.savefig("plots/delta_distr.png", bbox_inches='tight')
# %%
