import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib import cm
from utilities.mappings import *
import numpy as np
import pandas as pd

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def grouped_barplot_benchmark(df0, error_df=None, include_ui=False, subscript=None, ylabel=None, xlabel=None, ylim=None, title=None,
                              value_col='mean', rot=45, fig_size=(10,5), filename=None, legend_prop=None):
    plt.rcParams["figure.figsize"] = fig_size

    for dim in df0.index:
        print(f'Plotting {dim}...')
        entries = pd.DataFrame(df0.loc[dim])

        ux_ux = entries.loc['ux/ux']
        entries = entries.drop('ux/ux')

        sx_sx = entries.loc['sx/sx']
        entries = entries.drop('sx/sx')

        if include_ui:
            ui_ui = entries.loc["ui/ui"]
            entries = entries.drop('ui/ui')

        entry_labels = list(entries.index)

        if error_df is not None:
            low_err, high_err = [], []
            for l in entry_labels:
                raw_val = df0.loc[dim][l]
                raw_low_err = error_df.loc[dim][l+'-']
                low_err.append(raw_val - raw_low_err)
                raw_high_err = error_df.loc[dim][l+'+']
                high_err.append(raw_high_err - raw_val)
            yerr = [
                low_err, # low
                high_err # high
            ]
        else:
            yerr = None

        ax = entries.plot(
            kind='bar',
            ylim=ylim,
            title=title,
            rot=rot,
            color='darkgray',
            yerr=yerr
        )

        xticks = list(ax.get_xticks())
        xticks = [xticks[0] - 1] + xticks + [xticks[-1] + 1]

        uxuxline, = ax.plot(xticks, [ux_ux] * len(xticks), '--', label='ux/ux')
        sxsxline, = ax.plot(xticks, [sx_sx] * len(xticks), ':', label='sx/sx')
        handles = handles=[uxuxline, sxsxline]
        if include_ui:
            uiuiline, = ax.plot(xticks, [ui_ui] * len(xticks), ':', label='ui/ui')
            handles.append(uiuiline)
        if error_df is not None:
            ax.fill_between(xticks, [error_df.loc[dim]['ux/ux-']]*len(xticks), [error_df.loc[dim]['ux/ux+']]*len(xticks), alpha=0.2)
            ax.fill_between(xticks, [error_df.loc[dim]['sx/sx-']]*len(xticks), [error_df.loc[dim]['sx/sx+']]*len(xticks), alpha=0.2)

        ax.legend(handles=handles)

        if filename is not None:
            plt.savefig(f"{filename}_{dim}.png", format="png", bbox_inches="tight")

    plt.show()

def cohesive_grouped_barplot_benchmark(df0, error_df=None, include_ui=False, subscript=None, ylabel=None, xlabel=None, ylim=None, title=None,
                              value_col='mean', rot=45, fig_size=(10,5), filename=None, legend_prop=None):
    plt.rcParams["figure.figsize"] = fig_size

    fig, ax = plt.subplots()

    start_x = 1

    dims = []
    for dim in df0.index:
        dims.append(dim)
        print(f'Plotting {dim}...')
        entries = pd.DataFrame(df0.loc[dim])

        sx_sx = entries.loc['sx/sx']
        entries = entries.drop('sx/sx')

        if include_ui:
            ui_ui = entries.loc["ui/ui"]
            entries = entries.drop('ui/ui')

        entry_labels = list(entries.index)

        if error_df is not None:
            low_err, high_err = [], []
            for l in entry_labels:
                raw_val = df0.loc[dim][l]
                raw_low_err = error_df.loc[dim][l+'-']
                low_err.append(raw_val - raw_low_err)
                raw_high_err = error_df.loc[dim][l+'+']
                high_err.append(raw_high_err - raw_val)
            yerr = [
                low_err, # low
                high_err # high
            ]
        else:
            yerr = None

        ax.bar(
            x=[start_x],
            height=entries.loc["ui/sx"],
            width=0.5,
            yerr=yerr,
            color='lightgray'
        )

        xticks = [start_x]
        xticks = [xticks[0] - 0.4] + xticks + [xticks[-1] + 0.4]

        sxsxline, = ax.plot(xticks, [sx_sx] * len(xticks), ':', label='Sur', color='blue', linewidth=2)
        handles = handles=[sxsxline]
        if include_ui:
            uiuiline, = ax.plot(xticks, [ui_ui] * len(xticks), ':', label='Stu', color='green', linewidth=2)
            handles.append(uiuiline)
        if error_df is not None:
            ax.fill_between(xticks, [error_df.loc[dim]['sx/sx-']]*len(xticks), [error_df.loc[dim]['sx/sx+']]*len(xticks), alpha=0.2)

        start_x += 1

    ax.set_xticklabels(['']+[f"{dimensions_transformer[l]}$_{subscript}$" for l in dims])

    ax.legend(handles=handles)

    if filename is not None:
        plt.savefig(f"{filename}.png", format="png", bbox_inches="tight")

    plt.show()

def grouped_barplot(df0, subscript, ylabel, xlabel, ylim, title=None, value_col='mean', rot=45, fig_size=(10,5),
                    filename=None, legend_prop=None):
    df = df0.reset_index()
    plt.rcParams["figure.figsize"] = fig_size

    df['lower'] = df[value_col] - df["CI low"]
    cilow = df.pivot(index='label', columns='bot', values='lower')
    df['upper'] = df["CI high"] - df[value_col]
    cihigh = df.pivot(index='label', columns='bot', values='upper')

    err = []
    for col in cilow:
        err.append([cilow[col].values, cihigh[col].values])

    df0 = df.pivot(index='label', columns='bot', values=value_col)
    ax = df0.plot(
        kind='bar',
        ylim=ylim,
        title=title,
        rot=rot,
        yerr=err,
        color=[graphing_bot_colors[bot] for bot in df0.columns]
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) == 4: # 4 bots
        order = [0,2,1,3]
    else:
        order = range(len(labels))
    plt.legend([handles[i] for i in order], [bot_transformer.get(labels[i], labels[i]) for i in order], ncol=2, prop=legend_prop)
    # ax.get_legend().remove()
    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_xlabel(xlabel, labelpad=20)
    if subscript is not None:
        ax.set_xticklabels([dimensions_transformer[d]+f"$_{subscript}$" if d in dimensions_transformer else dimensions_transformer[d] for d in df0.index])
    else:
        ax.set_xticklabels([dimensions_transformer[d] if d in dimensions_transformer else dimensions_transformer[d] for d in df0.index])

    if filename is not None:
        plt.savefig(filename+'.png', format="png", bbox_inches="tight")

    plt.show()

def grouped_barplot_cohensd(df0, subscript, ylabel, xlabel, ylim=None, title=None, value_col='mean', rot=45,
                            fig_size=(10,5), filename=None, legend_prop=None, plot_err=True, width=1):
    df = df0.reset_index()
    plt.rcParams["figure.figsize"] = fig_size

    df['lower'] = df[value_col] - df["CI low"]
    cilow = df.pivot(index='label', columns='groups', values='lower')
    df['upper'] = df["CI high"] - df[value_col]
    cihigh = df.pivot(index='label', columns='groups', values='upper')

    err = []
    for col in cilow:
        err.append([cilow[col].values, cihigh[col].values])

    color_mapping = {
        "Stu$_i$/Sur$_x$": 'lightblue',
        "Stu$_i$/Stu$_x$": 'red',
        "Stu$_x$/Sur$_x$": 'green',
        "Stu$_x$": 'orange',
        "Sur$_x$": 'mediumpurple',
        "Dev$_x$/Stu$_x$": 'pink',
        "Dev$_x$/Stu$_i$": 'turquoise',
        "Dev$_x$/Sur$_x$": 'brown',
        "Dev$_x$": 'lightgray',
        "Stu$_i$": 'mediumspringgreen'
    }

    df0 = df.pivot(index='label', columns='groups', values=value_col)
    df0.columns = [c[2:] if '|' in c else c for c in df0.columns]
    ax = df0.plot(
        kind='bar',
        ylim=ylim,
        title=title,
        rot=rot,
        yerr=err if plot_err else None,
        color=[color_mapping[c] for c in df0.columns],
        width=width
    )

    # ax.get_legend().remove()
    ax.legend(ncol=len(df0.columns), loc='upper center')
    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_xlabel(xlabel, labelpad=0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if subscript:
        ax.set_xticklabels([dimensions_transformer[d]+f"$_{subscript}$" if d in dimensions_transformer else dimensions_transformer[d] for d in df0.index])
    else:
        ax.set_xticklabels([dimensions_transformer[d] if d in dimensions_transformer else dimensions_transformer[d] for d in df0.index])

    if filename is not None:
        plt.savefig(filename+'.png', format="png", bbox_inches="tight")

    plt.show()

def grouped_barplot_effectsize(df0, subscript, ylabel, xlabel, ylim=None, title=None, value_col='mean', rot=45,
                            fig_size=(10,5), filename=None, legend_prop=None):
    df = df0.reset_index()
    plt.rcParams["figure.figsize"] = fig_size

    df['lower'] = df[value_col] - df["CI low"]
    cilow = df.pivot(index='label', columns='method', values='lower')
    df['upper'] = df["CI high"] - df[value_col]
    cihigh = df.pivot(index='label', columns='method', values='upper')

    err = []
    for col in cilow:
        err.append([cilow[col].values, cihigh[col].values])

    df0 = df.pivot(index='label', columns='method', values=value_col)
    ax = df0.plot(
        kind='bar',
        ylim=ylim,
        title=title,
        rot=rot,
        yerr=err,
        color=['red', 'green']
    )

    # ax.get_legend().remove()
    # ax.set_ylabel(ylabel, labelpad=20)
    ax.set_xlabel("", labelpad=0)
    ax.set_xticklabels([dimensions_transformer[d] if d in dimensions_transformer else dimensions_transformer[d] for d in df0.index])

    if filename is not None:
        plt.savefig(filename+'.png', format="png", bbox_inches="tight")

    plt.show()

def grouped_barplot_3d(df):
    result = [df.xs('Emora', level='bot')['int mean'].tolist(),
              df.xs('Emora', level='bot')['ext mean'].tolist()]

    result = np.array(result, dtype=np.double)
    fig = plt.figure(figsize=(8, 8), dpi=250)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('Label', labelpad=10)
    ax1.set_ylabel('Group', labelpad=10)
    ax1.set_zlabel('Rating')
    xlabels = np.array(df.index.unique('label').tolist())
    xpos = np.arange(xlabels.shape[0])
    # xpos = np.arange(xlabels.shape[0]*df.index.unique('bot').size)
    ylabels = np.array(['Ui', 'Sx'])
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = result
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos

    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.w_yaxis.set_ticklabels(ylabels)

    values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
    plt.show()

def bot_barplot_3d(df):
    result = [df.xs('BART-FiD-RAG', level='bot')['int mean'].tolist(),
              df.xs('Blender-Decode', level='bot')['int mean'].tolist(),
              df.xs('Emora', level='bot')['int mean'].tolist(),
              df.xs('Blender2', level='bot')['int mean'].tolist()
              ]

    result = np.array(result, dtype=np.double)
    fig = plt.figure(figsize=(8, 8), dpi=250)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('Label', labelpad=10)
    ax1.set_ylabel('Bot', labelpad=10)
    ax1.set_zlabel('Rating')
    xlabels = np.array([dimensions_transformer[l] for l in df.index.unique('label').tolist()])
    xpos = np.arange(xlabels.shape[0])
    # xpos = np.arange(xlabels.shape[0]*df.index.unique('bot').size)
    ylabels = np.array(['BART-FiD-RAG', 'Blender-Decode', 'Emora', 'Blender2', ])
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = result
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos

    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.w_yaxis.set_ticklabels(ylabels)

    values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
    plt.show()

def plot_comparative(df0, subscript, title, value_col, fig_size, ylim=None, legend=True, filename=None):
    # plt.rc('legend', fontsize=10)  # legend fontsize

    # https://stackoverflow.com/questions/59922701/pandas-how-can-i-group-a-stacked-bar-chart
    plt.rcParams["figure.figsize"] = fig_size

    df0['lower'] = df0[value_col] - df0["CI low"]
    df0['upper'] = df0["CI high"] - df0[value_col]

    errLow = df0[['lower']].reset_index(['bot', 'label']).pivot(index='label', columns='bot', values='lower')
    errHi = df0[['upper']].reset_index(['bot', 'label']).pivot(index='label', columns='bot', values='upper')

    # 4 x 2 x 8 (bots x low, hi x labels)
    err = []
    for col in errLow:
        err.append([errLow[col].values, errHi[col].values])

    df0 = df0.unstack(level=-1)
    fig, ax = plt.subplots()

    bot_order = ['BART-FiD-RAG', 'Blender-Decode', 'Blender2', 'Emora']
    ordered_columns = [('win', b) for b in bot_order] + [('tie', b) for b in bot_order] + [('CI low', b) for b in bot_order] + [('CI high', b) for b in bot_order]
    df0 = df0[ordered_columns]

    groups = []
    for i in df0.columns:
        if i[1] not in groups:
            groups.append(i[1])

    # (df0['win']+df0['tie']+df0['lose']).plot(kind='bar', color=[graphing_bot_colors[i] for i in groups], alpha=0.2, rot=0, ax=ax)
    (df0['win']+df0['tie']).plot(kind='bar', color=[graphing_bot_colors[i] for i in groups], alpha=0.2, rot=0, ax=ax)
    df0['win'].plot(kind='bar', color=[graphing_bot_colors[i] for i in groups], rot=0, ax=ax, yerr=err)

    # h, l = ax.get_legend_handles_labels()
    # markers = {}
    # for h, l, (wtl, bot) in zip(h, l, df0.columns):
    #     markers.setdefault(bot, []).append((h,l))
    # wtl_dummies = [plt.plot([],marker="", ls="")[0]]*3 #4
    # bot_dummies = [plt.plot([],marker="", ls="")[0]]*4
    # handles = wtl_dummies
    # # labels = ["", "Lose:", "Tie:", "Win:"]
    # labels = ["", "Tie", "Win"]
    # for i, (bot, symbols) in enumerate(markers.items()):
    #     handles.append(bot_dummies[i])
    #     labels.append(bot_transformer.get(bot, bot))
    #     handles.extend([s[0] for s in symbols])
    #     labels.extend(["" for s in symbols])
    # leg = plt.legend(handles, labels, ncol=5, bbox_to_anchor=(0.80, 1.40), labelspacing=0.25)
    # for i, vpack in enumerate(leg._legend_handle_box.get_children()):
    #     if i == 0: # row titles
    #         for hpack in vpack.get_children():
    #             hpack.get_children()[0].set_width(0)
    #     else:
    #         for j, hpack in enumerate(vpack.get_children()):
    #             if j > 0: # bot win/tie/lose markers
    #                 hpack.get_children()[0].get_children()[0].set_width(30)
    #             else: # column titles
    #                 hpack.get_children()[0].set_width(0)

    h, l = ax.get_legend_handles_labels()
    plt.legend(h[-4:], l[-4:], ncol=4, loc='upper center')

    # ax.set_title(title)
    # ax.set_ylabel('Proportion', labelpad=20)
    # ax.set_xlabel('Label', labelpad=20)
    ax.set_xlabel('')
    if subscript is not None:
        ax.set_xticklabels([dimensions_transformer[d]+f"$_{subscript}$" for d in df0.index])
    else:
        ax.set_xticklabels([dimensions_transformer[d] for d in df0.index])
    if ylim:
        ax.set_ylim(ylim)

    if not legend:
        ax.get_legend().remove()

    if filename is not None:
        plt.savefig(filename+'.png', format="png", bbox_inches="tight")

    plt.tight_layout()
    plt.show()
    return df0

def scatter_graph_with_error(df, column, ylabel, overlay=None, significance=None, ylim=(None, None),
                             figsize=(20, 5), rot=None, xaxis_colored=True, filename=None, single_color=None, fmt='o', ax=None):

    new_ax = False
    if ax is None:
        plt.rcParams["figure.figsize"] = figsize
        fig, ax = plt.subplots()
        new_ax = True

    def plot_by_category(ax, df, category, color, xaxis_start, width=1, transform=None):
        extracted = df[df["category"] == category]
        subscript = category.split(" ")[-1][0]
        labels = [dimensions_transformer[l] + f'$_{subscript}$' for l in extracted["label"]]
        attributes = [dict(facecolor="lightyellow", alpha=0.7) if significance is not None and significance.xs((c,l))['pvalue'] < 0.05 else {} for c, l in zip(extracted["category"],extracted["label"])]
        lower_bound = extracted[column] - extracted["CI low"]
        upper_bound = extracted["CI high"] - extracted[column]
        xaxis_end = xaxis_start + len(extracted)
        ax.errorbar(np.arange(xaxis_start, xaxis_end),
                    extracted[column],
                    yerr=[lower_bound, upper_bound],
                    fmt=fmt,
                    elinewidth=width,
                    color=color,
                    transform=transform)
        if None not in ylim:
            ax.set_ylim(*ylim)
        return labels, attributes, xaxis_end

    if single_color:
        likert_dialogue_color = single_color
        comparative_color = single_color
    else:
        likert_dialogue_color = "red"
        comparative_color = "green"

    flattened_df = df.reset_index()
    width = 2

    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData if overlay is not None else None
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData if overlay is not None else None

    likert_dialogue_labels, likert_dialogue_attributes, comparative_start = plot_by_category(ax, flattened_df, "likert dialogue",
                                                                 likert_dialogue_color, 0, width=width, transform=trans1)
    comparative_labels, comparative_attributes, misc_start = plot_by_category(ax, flattened_df, "comparative", comparative_color,
                                                      comparative_start, width=width, transform=trans1)

    if overlay is not None:
        overlay_agreements = overlay.reset_index()
        _, _, _ = plot_by_category(ax, overlay_agreements, "likert dialogue", "gray", 0, transform=trans2)
        _, _, _ = plot_by_category(ax, overlay_agreements, "comparative", "gray", comparative_start, transform=trans2)

    category_range = {comparative_start: likert_dialogue_color, misc_start: comparative_color}
    xaxis_colors = {}
    prev_idx = 0
    for idx, color in category_range.items():
        for i in range(prev_idx, idx):
            xaxis_colors[i] = color
        prev_idx = idx

    ax.set_ylabel(ylabel, labelpad=20, rotation=0)
    xpos = np.arange(len(flattened_df))
    ax.set_xticks(xpos)
    all_attributes = likert_dialogue_attributes + comparative_attributes
    ax.set_xticklabels(likert_dialogue_labels + comparative_labels, rotation=rot)
    for i, (tickloc, ticklabel) in enumerate(zip(ax.get_xticks(), ax.get_xticklabels())):
        if xaxis_colored:
            ticklabel.set_color(xaxis_colors[tickloc])
        if all_attributes[i]:
            ticklabel.set_bbox(all_attributes[i])

    if new_ax:
        # ax.yaxis.grid(True)
        # Save the figure and show
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename+'.png', format="png", bbox_inches="tight")
        plt.show()

def plot_predictive_validity(rsquareds, sorted=None):
    # Build the plot
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 28

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=28)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams["figure.figsize"] = (30, 10)

    rsquareds_toplot = rsquareds
    if sorted is not None:
        sorted_rsquareds = rsquareds.sort_values(by=sorted, ascending=[True, False])
        rsquareds_toplot = sorted_rsquareds

    ax = rsquareds_toplot.plot(
        kind='bar',
        width=0.7,
        color=['lightgray', 'dimgray', 'black'],
        edgecolor='black',
        linewidth=1,
        ylim=(0, 0.11)
    )

    # bars = ax.patches
    # patterns = ['', '/']
    # hatches = [p for p in patterns for i in range(len(just_rsquareds))]
    # for bar, hatch in zip(bars, hatches):
    #     bar.set_hatch(hatch)

    ax.legend()

    xticklabels = ax.get_xticklabels()

    def transform_label(item):
        evaluation_cat, label = item.get_text()[1:-1].split(',')
        subscript = evaluation_cat.replace("likert ", "")[0]
        return dimensions_transformer[label.strip()] + f"$_{subscript}$"

    abbrev_xticklabels = [transform_label(xtl) for xtl in xticklabels]

    likert_turn_color = "blue"
    likert_dialogue_color = "red"
    comparative_color = "green"
    behavior_color = "darkorange"

    category_range = {16: behavior_color, 23: likert_turn_color, 30: likert_dialogue_color, 37: comparative_color}
    xaxis_colors = {}
    prev_idx = 0
    for idx, color in category_range.items():
        for i in range(prev_idx, idx):
            xaxis_colors[i] = color
        prev_idx = idx

    ax.set_xticklabels(abbrev_xticklabels, rotation=90)
    for tickloc, ticklabel in zip(plt.gca().get_xticks(), plt.gca().get_xticklabels()):
        ticklabel.set_color(xaxis_colors[tickloc])

    ax.set_ylabel('R$^2$', rotation=0, labelpad=40)
    ax.set_xlabel('')

    plt.tight_layout()
    plt.show()