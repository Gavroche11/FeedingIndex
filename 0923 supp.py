#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.font_manager as fm
from matplotlib.image import NonUniformImage
import cv2

from glob import glob

from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


def successive(frames):
    result = list()
    i = 0
    if len(frames) == 0:
        return list()
    else:
        while i in range(len(frames)):
            if i == 0:
                newlist = [frames[i]]
            elif frames[i] - frames[i - 1] == 1:
                newlist.append(frames[i])
            else:
                result.append(newlist)
                newlist = [frames[i]]
            i += 1
        result.append(newlist)
        return result

def representation(frames, x=1):
    assert x == 0 or x == 1 or x == 2, 'Input 0, 1, or 2 for first, middle, or last frame respectively'
    sf = successive(frames)
    result = list()
    for i, j in enumerate(sf):
        if x == 0:
            result.append(j[0])
        elif x == 1:
            result.append(round((j[0] + j[-1]) / 2))
        else:
            result.append(j[-1])
    return result

def frequency(arr, length, FPS):
    def make_window(frame):
        return np.linspace(frame, frame + length, num=length, endpoint=False, dtype=int)
    def count(window):
        return arr[window].sum(axis=0)
    indices = np.arange(arr.shape[0] - length + 1).reshape(-1, 1)
    return np.apply_along_axis(count, 1, np.apply_along_axis(make_window, 1, indices)) / (length / FPS)

def check_backwards(indices_for_eval, target, length):
    '''
    indices for eval: 1D array
    target: 1D array TF for whole frames
    '''
    def make_window(frame):
        return np.linspace(frame, frame - length + 1, num=length, endpoint=True, dtype=int)
    indices_for_eval = indices_for_eval[indices_for_eval >= length - 1].reshape(-1, 1)
    back = target[np.apply_along_axis(make_window, 1, indices_for_eval)].any(axis=1).reshape(-1)

    front = []
    for i in indices_for_eval[indices_for_eval < length -1]:
        front.append(target[np.arange(i+1, dtype=int)].any())
    front = np.array(front)

    return indices_for_eval[np.concatenate((front, back)).astype(bool)].reshape(-1)

def lininterpol(df, bodyparts, ll_crit, absolute=True):
    '''
    df: DataFrame from pd.read_csv(filepath, header=[1, 2], index_col=0, skiprows=0)
    bodyparts: list of bodyparts of interest
    ll_crit: real number in [0, 1) or list of real numbers
    absolute: bool or list of bools
    '''
    numbodyparts = len(bodyparts)
    numframes = len(df)

    values = df[bodyparts].values.reshape(-1, numbodyparts, 3).transpose([1, 0, 2]) # np array of shape (numbodyparts, numframes, 3)
    
    if type(absolute) == bool:
        absolute = [absolute] * numbodyparts
    
    if type(ll_crit) == float:
        ll_crit = [ll_crit] * numbodyparts

    mins = []
    for i in range(numbodyparts):
        if absolute[i]:
            mins.append(ll_crit[i])
        else:
            cutoff_index = values[i, :, -1].argsort()[int(numframes * ll_crit[i])]
            mins.append(values[i, cutoff_index, -1])

    mins = np.array(mins).reshape(-1, 1) # np array of shape (numbodyparts, 1)
    
    good = values[:, :, -1] >= mins # np array of shape (numbodyparts, numframes), T or F

    assert good.all(axis=0).sum() >= 2, 'Likelihood too high'

    start, end = np.where(good.all(axis=0))[0][0], np.where(good.all(axis=0))[0][-1]

    values = values[:, start:(end + 1), :] # np array of shape (numbodyparts, # frames for use, 3)
    good = good[:, start:(end + 1)] # np array of shape (numbodyparts, # frames for use)

    for i in range(numbodyparts):
        bad0 = np.array(representation(np.where(~good[i])[0], x=0)).reshape(-1, 1)
        bad1 = np.array(representation(np.where(~good[i])[0], x=2)).reshape(-1, 1)
        bads = np.concatenate((bad0, bad1), axis=1)

        for j in range(bads.shape[0]):
            prev_frame = int(bads[j, 0] - 1)
            next_frame = int(bads[j, 1] + 1)
            values[i, prev_frame:next_frame, :-1] = np.linspace(values[i, prev_frame, :-1], values[i, next_frame, :-1], num=(next_frame - prev_frame), endpoint=False)

    tuples = []
    for bp in bodyparts:
        tuples.append((bp, 'x'))
        tuples.append((bp, 'y'))


    new_df = pd.DataFrame(values[:, :, :-1].transpose([1, 0, 2]).reshape(-1, 2 * numbodyparts), columns=pd.MultiIndex.from_tuples(tuples))

    '''
    returns:
        (new_df of index=df.index, columns consisting of (bodypart, x) and (bodypart, y))
        start,
        end
    '''
    return new_df, start, end

def return_bout(filepath, params, ll_crit=0.7, absolute=True, latency=0.5, FPS=30, interval=0.5):
    '''
    params: (trayx, trayy, dist0, dist1)
        dist0: mouth-hand
        dist1: hand-tray
    '''

    coords = pd.read_csv(filepath, header=[1, 2], index_col=0, skiprows=0).reset_index()
    coords, start, end = lininterpol(coords, ['Mouth', 'R_hand', 'L_hand'], ll_crit=ll_crit, absolute=absolute)

    trayx, trayy, dist0, dist1 = params

    coords['Condition 0R'] = np.sqrt(np.square(coords[[('Mouth', 'x'), ('Mouth', 'y')]].values - coords[[('R_hand', 'x'), ('R_hand', 'y')]].values).sum(axis=1)) < dist0
    coords['Condition 0L'] = np.sqrt(np.square(coords[[('Mouth', 'x'), ('Mouth', 'y')]].values - coords[[('L_hand', 'x'), ('L_hand', 'y')]].values).sum(axis=1)) < dist0
    coords['Condition 1R'] = np.sqrt(np.square(coords[[('R_hand', 'x'), ('R_hand', 'y')]].values - np.array([trayx, trayy]).reshape(1, -1)).sum(axis=1)) < dist1
    coords['Condition 1L'] = np.sqrt(np.square(coords[[('L_hand', 'x'), ('L_hand', 'y')]].values - np.array([trayx, trayy]).reshape(1, -1)).sum(axis=1)) < dist1

    R_old = representation(coords[coords['Condition 0R']].index, x=0)
    R_new = []
    for frame in R_old:
        if len(R_new) == 0:
            R_new.append(frame)
        elif frame - R_new[-1] > FPS * interval:
            R_new.append(frame)
    
    L_old = representation(coords[coords['Condition 0L']].index, x=0)
    L_new = []
    for frame in L_old:
        if len(L_new) == 0:
            L_new.append(frame)
        elif frame - L_new[-1] > FPS * interval:
            L_new.append(frame)
    

    coords['Bout R'] = pd.Series(True, index=check_backwards(np.array(R_new), coords['Condition 1R'].values, int(latency * FPS)))
    coords['Bout L'] = pd.Series(True, index=check_backwards(np.array(L_new), coords['Condition 1L'].values, int(latency * FPS)))
    coords['Bout R'] = coords['Bout R'].fillna(False)
    coords['Bout L'] = coords['Bout L'].fillna(False)

    coords['Bout'] = (coords['Bout R'] | coords['Bout L'])

    return coords, start, end

fz = pd.read_csv('/content/drive/Shareddrives/FNMR Deeplabcut/Monkey/Excel/R1019파, R1047빨, R1051초/coordinate(excel)/coords_foodzone.csv',
                 header=None, index_col=0)
fz_dict = dict()
for i in fz.index:
    clean = i.replace('C:/Users/User/Documents/GOMPlayer/Capture/210819\\', '')
    clean = clean[:clean.index('sDLC')]
    spl = clean.split('_R')
    if spl[0][-1] == '1':
        exp = 'Artificial cycle 1'
    elif spl[0][-1] == '2':
        exp = 'Artificial cycle 2'
    else:
        exp = 'Natural'
    
    case = (exp, spl[0][:2], spl[1])

    fz_dict[case] = fz.loc[i, 1]


# In[ ]:


colors_dict = {
    '1019': 'blue',
    '1047': 'red',
    '1051': 'green'
}

exp_list = ['Natural', 'Artificial cycle 1', 'Artificial cycle 2']
state_list = ['FP', 'FU', 'SP', 'SU']
mon_list = ['1019', '1047', '1051']

cases = []
for i in exp_list:
    for j in state_list:
        for k in mon_list:
            cases.append((i, j, k))

files_dict = dict()
for case in cases:
    exp, state, mon = case
    folder = '/content/drive/Shareddrives/FNMR Deeplabcut/Monkey/Excel/R1019파, R1047빨, R1051초/{}/{}/'.format(exp, state)
    csv_list = glob(folder + '*.csv')
    for csv_name in csv_list:
        if mon in csv_name.replace(folder, ''):
            files_dict[case] = csv_name
            break

params_dict = dict()

trayhand = pd.read_csv('/content/drive/MyDrive/FNMR/0819/trayhand.csv', header=None, index_col=0)
for i in trayhand.index:
    clean = i.replace('C:/Users/User/Documents/GOMPlayer/Capture/210819\\', '')
    clean = clean[:clean.index('sDLC')]
    spl = clean.split('_R')
    if spl[0][-1] == '1':
        exp = 'Artificial cycle 1'
    elif spl[0][-1] == '2':
        exp = 'Artificial cycle 2'
    else:
        exp = 'Natural'
    
    case = (exp, spl[0][:2], spl[1])

    cur_values = trayhand.loc[i].values

    params_dict[case] = (int(cur_values[0]), int(cur_values[1]), 32, np.sqrt(np.square(cur_values[:2] - cur_values[2:]).sum()))

processed_files_dict = dict()

for case in cases:
    processed_files_dict[case] = '/content/drive/MyDrive/FNMR/absolute 0.7/{}_{}_{}.csv'.format(case[0], case[1], case[2])


# In[ ]:


get_ipython().system("wget 'https://www.freefontspro.com/d/14454/arial.zip'")
get_ipython().system("unzip 'arial.zip'")
get_ipython().system("mv '/content/arial.ttf' '/usr/share/fonts/truetype'")

def arial_fontprop(size, weight):
    return fm.FontProperties(fname='/usr/share/fonts/truetype/arial.ttf', weight=weight, size=size)


# In[ ]:


latency = 3
FPS = 30
interval = 0.5
freq_length = 5
ll_crit=0.7
absolute=True


# Figure 5 A-B
# 
# Approach
# 
# Artificial cycle 1, 1051, FP / SU
# 
# 4x2
# 
# row 1, 2: min of R_dist, L_dist
# 
# threshold 표시
# 
# row 3, 4: Appr binary / freq

# In[ ]:


# ?????


temp_sl = ['FP', 'SU']

width = 0.02

labelprop = arial_fontprop(35, 'bold')
tickprop = arial_fontprop(35, 'bold')
legendprop = arial_fontprop(20, 'medium')
titleprop = arial_fontprop(60, 'bold')

for exp in exp_list:
    for mon in mon_list:
        fig = plt.figure(figsize=(80, 50), constrained_layout=True)
        grid0 = plt.GridSpec(1, 2)
        for j, state in enumerate(temp_sl):
            fake = fig.add_subplot(grid0[j])
            if state == 'FP':
                newstate = 'HI-PF'
            else:
                newstate = 'SI-UF'
            
            if mon == '1019':
                newmon = 'Monkey A'
            elif mon == '1047':
                newmon = 'Monkey B'
            else:
                nemwon = 'Monkey C'
            
            fake.set_title(newstate+'\n\n', fontproperties=titleprop)
            fake.set_axis_off()
            grid1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid0[j], height_ratios=[5, 2])
    
            case = (exp, state, mon)
            filepath = files_dict[case]
            params = params_dict[case]

            df = pd.read_csv(filepath, index_col=0, header=[1, 2], skiprows=0)
            coords, _, _ = lininterpol(df, ['R_hand', 'L_hand'], ll_crit=ll_crit, absolute=absolute)
                
            trayx, trayy, dist1 = params_dict[case][0], params_dict[case][1], params_dict[case][3]

            dist_R = np.sqrt(np.square(coords['R_hand'].values - np.array([[trayx, trayy]])).sum(axis=1))
            dist_L = np.sqrt(np.square(coords['L_hand'].values - np.array([[trayx, trayy]])).sum(axis=1))
            dist_min = np.vstack((dist_R, dist_L)).min(axis=0)

            coords['R_in'] = dist_R < dist1
            coords['L_in'] = dist_L < dist1

            R_old = representation(coords[coords['R_in']].index, x=0)
            R_new = []
            for frame in R_old:
                if len(R_new) == 0:
                    R_new.append(frame)
                elif frame - R_new[-1] > FPS * interval:
                    R_new.append(frame)

            L_old = representation(coords[coords['L_in']].index, x=0)
            L_new = []
            for frame in L_old:
                if len(L_new) == 0:
                    L_new.append(frame)
                elif frame - L_new[-1] > FPS * interval:
                    L_new.append(frame)

            result_indices = np.array(list(set(R_new + L_new)))
            result_indices = result_indices[result_indices.argsort()]
                
            coords['Approach'] = pd.Series(True, index=result_indices)
            coords['Approach'] = coords['Approach'].fillna(False)

            apprs = coords[coords['Approach']].index.to_numpy()
            start = []
            end = []
            for appr in apprs:
                if len(start) == 0:
                    start.append(appr)
                    end.append(appr + width * FPS * 60)
                elif appr > end[-1]:
                    start.append(appr)
                    end.append(appr + width * FPS * 60)
                else:
                    end[-1] = appr + width * FPS * 60
            start = np.array(start)
            end = np.array(end)
            bins = np.vstack((start, end)).T.reshape(-1)

            freq = frequency(coords['Approach'].values, FPS * freq_length, FPS).reshape(-1)
            freq = np.concatenate((freq, np.full(shape=(FPS * freq_length - 1,), fill_value=freq[-1])))
    
            time = coords.index / (FPS*60)
    
            # dist avg
            grid_R = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid1[0], height_ratios=[3, 2])
            # row 1
            if j == 0:
                ax = fig.add_subplot(grid_R[0])
                ax_ref0 = ax
                ax.set_yticks(np.linspace(0, 1200, num=7, endpoint=True, dtype=int))
            else:
                ax = fig.add_subplot(grid_R[0], sharex=ax_ref0, sharey=ax_ref0)

            ax.plot(time, dist_min, color='black', label=newmon)
            ax.bar([0], [dist1], width=20, align='edge', color=(1, 1, 0.6, 0.7), edgecolor='yellow', linewidth=1, label='Within Threshold')
            
            ax.set_xticks(np.arange(21))
            ax.set_xlim([0, 20])
            ax.set_xticklabels(ax.get_xticks(), fontproperties=tickprop)
            ax.set_yticklabels(ax.get_yticks(), fontproperties=tickprop)
            ax.set_xlabel('Time (min.)', fontproperties=labelprop, labelpad=8)
            ax.set_ylabel('Distance (px)', fontproperties=labelprop)
            ax.legend(loc='upper right', prop=legendprop)

            # row 2
            ax = fig.add_subplot(grid_R[1], sharex=ax_ref0)
            
            ax.plot(time, dist_min, color='black', label=newmon)
            ax.bar([0], [dist1], width=20, align='edge', color=(1, 1, 0.6, 0.7), edgecolor='yellow', linewidth=1, label='Within Threshold')

            ax.set_xticklabels(ax.get_xticks(), fontproperties=tickprop)
            ax.set_xlabel('Time (min.)', fontproperties=labelprop, labelpad=8)

            ax.set_yticks([0, dist1, 200])
            ax.set_ylim([0, 200])
            ax.set_yticklabels(['0.0', str(np.around(dist1, decimals=1))+'*', '200.0'], fontproperties=tickprop)
            
            ax.set_ylabel('Distance (px)', fontproperties=labelprop)
            ax.legend(loc='upper right', prop=legendprop)


            # Freq
            grid_F = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid1[1], height_ratios=[1, 3], hspace=0)
            ax = fig.add_subplot(grid_F[0], sharex=ax_ref0)
            
            ax.hist(start / (FPS*60), bins=bins / (FPS*60), label=newmon, color='black')
            
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.set_ylim([0, 1])
            ax.set_yticks([0., 1.])
            ax.set_yticklabels(ax.get_yticks(), fontproperties=tickprop, color=(1, 1, 1, 1))
            ax.set_ylabel('Approach', fontproperties=labelprop)
            ax.legend(loc='upper right', prop=legendprop)

            if j == 0:
                ax = fig.add_subplot(grid_F[1], sharex=ax_ref0)
                ax_ref1 = ax
                ax.set_ylim([0, 2])
                ax.set_yticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
            else:
                ax = fig.add_subplot(grid_F[1], sharex=ax_ref0, sharey=ax_ref1)
            
            ax.plot(time, freq, label='Monkey C', color='black')
            
            ax.set_xticklabels(ax.get_xticks(), fontproperties=tickprop)
            ax.set_yticklabels(ax.get_yticks(), fontproperties=tickprop)
            ax.set_xlabel('Time (min.)', fontproperties=labelprop, labelpad=8)
            ax.set_ylabel('Approach Frequency ($\mathregular{s^{-1}}$)', fontproperties=labelprop)
            ax.legend(loc='upper right', prop=legendprop)

        fig.suptitle('{} {}\n'.format(exp, newmon), fontproperties=titleprop)
        
        fig.savefig('/content/drive/MyDrive/FNMR/0923/Appr/Appr_{}_{}.png'.format(exp, mon))
        plt.close()


# Traj

# In[12]:


tmp_cases = [('Artificial cycle 1', 'FP', '1019'),
             ('Artificial cycle 1', 'FP', '1051'),
             ('Artificial cycle 1', 'SU', '1019'),
             ('Artificial cycle 1', 'SU', '1051')]

bodyparts = ['Head_top', 'Mouth', 'Body_center']

bp1_color_last = np.array([255, 24, 128])
bp4_color_last = np.array([160, 240, 165])
bp5_color_last = np.array([253, 111, 53])

color_array = np.vstack((bp1_color_last, bp4_color_last, bp5_color_last))
color_array = color_array[:, ::-1] / 255
color_array = np.concatenate((color_array, np.full(shape=(3, 1), fill_value=1.0)), axis=1)

labelprop = arial_fontprop(50, 'bold')
tickprop = arial_fontprop(40, 'bold')
titleprop = arial_fontprop(60, 'bold')
legendprop = arial_fontprop(40, 'medium')


for case in tmp_cases:
    filepath = files_dict[case]
    df = pd.read_csv(filepath, header=[1, 2], skiprows=0, index_col=0)
    video_path = filepath.replace('.csv', '_labeled.mp4')
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if case[1] == 'FP':
        title = case[0] + ' HI-PF\n'
    else:
        title = case[0] + ' SI-UF\n'

    if case[-1] == '1019':
        title += 'Monkey A'
    else:
        title += 'Monkey C'
    
    fig = plt.figure(figsize=(width/20, height/20))
    coords, _, _ = lininterpol(df, bodyparts, ll_crit=ll_crit, absolute=absolute)
    ax = plt.gca()
    for j, bp in enumerate(bodyparts):
        # Scatter
        ax.scatter(coords.values[:9000, 2*j], coords.values[:9000, 2*j+1], s=2, c=color_array[[j], :], label=bp)
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    ax.legend(loc='upper right', prop=legendprop)
    ax.invert_yaxis()
    ax.set_xticklabels(ax.get_xticks().astype(int), fontproperties=tickprop)
    ax.set_yticklabels(ax.get_yticks().astype(int), fontproperties=tickprop)
    ax.set_xlabel('x (px)', fontproperties=labelprop)
    ax.set_ylabel('y (px)', fontproperties=labelprop)
    fig.suptitle('{}\n\n'.format(title), fontproperties=titleprop)
    fig.savefig('/content/drive/MyDrive/FNMR/0923/Traj/{}_scatter_withoutbg.png'.format(title.replace('\n', ' ')))
    plt.close()

    fig = plt.figure(figsize=(width/20, height/20))
    ax = plt.gca()
    background = plt.imread('/content/drive/Shareddrives/FNMR Deeplabcut/Monkey/그래프들/cage background/{}1_R{}_cage_background.png'.format(case[1], case[2])) # cage path
    ax.imshow(background)
    for j, bp in enumerate(bodyparts):
        # Scatter
        ax.scatter(coords.values[:9000, 2*j], coords.values[:9000, 2*j+1], s=2, c=color_array[[j], :], label=bp)
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    ax.legend(loc='upper right', prop=legendprop)
    ax.invert_yaxis()
    ax.set_axis_off()
    fig.suptitle('{}\n\n'.format(title), fontproperties=titleprop)
    fig.savefig('/content/drive/MyDrive/FNMR/0923/Traj/{}_scatter_withbg.png'.format(title.replace('\n', ' ')))
    plt.close()


    fig = plt.figure(figsize=(width/20, height/20))
    ax = plt.gca()
    ax.imshow(background)
    means = coords.values[:9000].reshape(-1, 30, 2*len(bodyparts)).mean(axis=1)
    for j, bp in enumerate(bodyparts):
        ax.scatter(means[:, 2*j], means[:, 2*j+1], s=100, c=color_array[[j], :], label=bp)
    for k in range(300):
        plt.plot(means[k, [0, 2]], means[k, [1, 3]], color=(1, 0, 0, 0.5), linewidth=2)
        plt.plot(means[k, [2, 4]], means[k, [3, 5]], color=(1, 0, 0, 0.5), linewidth=2)
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    ax.legend(loc='upper right', prop=legendprop)
    ax.invert_yaxis()
    ax.set_axis_off()
    fig.suptitle('{}\n\n'.format(title), fontproperties=titleprop)
    fig.savefig('/content/drive/MyDrive/FNMR/0923/Traj/{}_line_withbg.png'.format(title.replace('\n', ' ')))
    plt.close()


# In[ ]:




