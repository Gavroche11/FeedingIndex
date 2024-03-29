import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from matplotlib.image import NonUniformImage

# Sample code related to Fig. 7, S10-11

def lininterpol(df, bodyparts, ll_crit, absolute=True):
    '''
    df: DataFrame from pd.read_csv(filepath, header=[1, 2], index_col=0, skiprows=0)
    bodyparts: list of bodyparts of interest
    ll_crit: real number in [0, 1) or list of real numbers in [0, 1)
        In the first case, the same value of ll_crit is applied to all bodyparts.
        In the second case, each value of ll_crit in the list is applied to corresponding bodyparts.
    absolute: bool or list of bools
        If absolute=True, the cutoff criterion is ll_crit itself.
        If absolute=False, the cutoff criterion is the int(# frames * ll_crit)-th lowest likelihood for each bodoypart.
        If absoulte is list, each boolean value in the list is applied to corresponding bodyparts.
    Returns:
    new_df: pd.DataFrame, bad values cut off and interpolated linearly.
    start, end: the indices in the original df corresponding to the first/last index of new_df.
        If df starts/ends with bad values, those are all cut off and not interpolated; so the first/last index of new_df may not represent the first index of df.
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

    assert good.all(axis=0).sum() >= 2, 'Likelihood criterion too high'

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

bodyparts = ['Head_top', 'Mouth', 'Body_center']

color_array = np.array([
                        [255, 24, 128],
                        [160, 240, 165],
                        [253, 111, 53]
]) # numpy array of shape (len(bps), 3) (Same as colors_last in Trajectory Video.py)

color_array = color_array[:, ::-1] / 255
color_array = np.concatenate((color_array, np.full(shape=(3, 1), fill_value=1.0)), axis=1)

filepath = 'DLC_OUTPUT_CSV_FILE_PATH' # Write the path of DLC output csv file
df = pd.read_csv(filepath, header=[1, 2], skiprows=0, index_col=0)
video_path = 'VIDEO_PATH' # Write the path of the corresponding video file
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
background_path = 'CAGE_BACKGROUND' # Write the path of background image

# Fig. 7A, S10
top = cm.get_cmap('jet', 256)
newcmp = matplotlib.colors.ListedColormap(np.concatenate((np.zeros(shape=(10, 4)), top(np.linspace(0, 1, 256))), axis=0),
                                          name='newjet')

for bp in bodyparts:
    coords, _, _ = lininterpol(df, [bp], ll_crit=ll_crit, absolute=absolute)
    raw = coords[[(bp, 'x'), (bp, 'y')]].values

    num_rows = int(height / 4)
    num_cols = int(width / 4)
    H, xedges, yedges = np.histogram2d(raw[:, 0], raw[:, 1], bins=[num_cols, num_rows], range=[[0, width], [0, height]], density=True)
    H = H.T[::-1]

    background = plt.imread(background_path)
    fig, ax = plt.subplots(figsize=(width/20, height/20))
    ax.set_aspect('equal')
    ax.set_xlim(xedges[[0, -1]])
    ax.set_ylim(yedges[[0, -1]])

    background = background[::-1]
    ax.imshow(background)

    im = NonUniformImage(ax, interpolation='bilinear', cmap=newcmp,
                    extent=(0, width, 0, height),
                    norm=matplotlib.colors.Normalize(0, 0.0001),
                    )
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    im.set_data(xcenters, ycenters, H)
    ax.images.append(im)
    ax.set_axis_off()

# Fig. S11A-B top
fig = plt.figure(figsize=(width/20, height/20))
coords, _, _ = lininterpol(df, bodyparts, ll_crit=ll_crit, absolute=absolute)
ax = plt.gca()
for j, bp in enumerate(bodyparts):
    ax.scatter(coords.values[:9000, 2*j], coords.values[:9000, 2*j+1], s=2, c=color_array[[j], :])
    ax.scatter([], [], s=100, c=color_array[[j], :], label=bp)
ax.set_xlim([0, width])
ax.set_ylim([0, height])
ax.legend(loc='upper right')
ax.invert_yaxis()


# Fig. S11A-B middle
fig = plt.figure(figsize=(width/20, height/20))
ax = plt.gca()
background = plt.imread(background_path)
ax.imshow(background)
for j, bp in enumerate(bodyparts):
    ax.scatter(coords.values[:9000, 2*j], coords.values[:9000, 2*j+1], s=2, c=color_array[[j], :])
    ax.scatter([], [], s=100, c=color_array[[j], :], label=bp)
ax.set_xlim([0, width])
ax.set_ylim([0, height])
ax.legend(loc='upper right')
ax.invert_yaxis()
ax.set_axis_off()


# Fig. 7B, S11A-B Bottom
fig = plt.figure(figsize=(width/20, height/20))
ax = plt.gca()
background = plt.imread(background_path)
ax.imshow(background)
means = coords.values[:9000].reshape(-1, 30, 2*len(bodyparts)).mean(axis=1)
for j, bp in enumerate(bodyparts):
    ax.scatter(means[:, 2*j], means[:, 2*j+1], s=100, c=color_array[[j], :], label=bp)
for k in range(300):
    ax.plot(means[k, [0, 2]], means[k, [1, 3]], color=(1, 0, 0, 0.5), linewidth=2)
    ax.plot(means[k, [2, 4]], means[k, [3, 5]], color=(1, 0, 0, 0.5), linewidth=2)
ax.set_xlim([0, width])
ax.set_ylim([0, height])
ax.legend(loc='upper right')
ax.invert_yaxis()
ax.set_axis_off()
