import numpy as np
import pandas as pd

# Functions

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


def temporal_density(arr, time, FPS):
    '''
    arr: 1D array of True, False
    time: float, length of moving bin (s)
    FPS: FPS of the video
    '''
    length = int(time * FPS)
    def make_window(frame):
        return np.linspace(frame, frame + length, num=length, endpoint=False, dtype=int).T
    indices = np.arange(arr.shape[0] - length + 1)
    td = arr[make_window(indices).reshape(-1, length)].sum(axis=1) / time
    return np.concatenate((td, np.full(shape=(length - 1,), fill_value=td[-1])))

def check_backwards(indices_for_eval, target, length):
    '''
    indices_for_eval: 1D array of indices for evaluation
    target: 1D array of True, False for whole frames
    length: positive int
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

# D-AF (frequency of tray approach)

def return_approach(filepath, params, ll_crit=0.7, absolute=True, interval=0.5, FPS=30):
    '''
    filepath: output of DLC; its bodyparts must contain 'R_hand', 'L_hand'

    params: (
        x-coord of tray,
        y-coord of tray,
        threshold of the distance btwn hand and tray
    )

    ll_crit, absolute: look at lininterpol function for further description

    interval: nonnegative float. the minimum time interval (s) btwn two contiguous events of the same hand entering the circle of radius params[2] centered at tray. default 0.5.
        For example, suppose the right hand entered the circle of radius params[2] centered at tray at 1.0s.
        In case of interval=0.5, we do not record this event until 1.5s, even if it appears to happen according to the DLC coordinates.
        This is because it is unlikely that the same hand enters the circle more than twice in 0.5s; this is rather because the labeled coordinate is inaccurate for a few frames.
        Hence, we exclude those events happening multiple times in some small time interval, so that our calculation is more compatible with reality.

    FPS: FPS of the video. default 30

    Returns:
    coords: DataFrame. coords['Approach'] is the Series of boolean value whether each frame is 'approach' or not.
    start, end: look at lininterpol function for further description
    '''
    df = pd.read_csv(filepath, index_col=0, header=[1, 2], skiprows=0)
    coords, start, end = lininterpol(df, ['R_hand', 'L_hand'], ll_crit=ll_crit, absolute=absolute)

    trayx, trayy, dist1 = params

    coords['R_in'] = np.linalg.norm(coords['R_hand'].values - np.array([[trayx, trayy]]), axis=1) < dist1
    coords['L_in'] = np.linalg.norm(coords['L_hand'].values - np.array([[trayx, trayy]]), axis=1) < dist1
                                    
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

    return coords, start, end

# D-AD (duration in food zone)

def return_infz(filepath, fz_x, ll_crit=0.7, absolute=True):
    '''
    filepath: output of DLC; its bodyparts must contain 'Mouth', 'R_hand', 'L_hand'

    fz_x: x-coord of food zone
    
    ll_crit, absolute: look at lininterpol function for further description

    Returns:
    coords: DataFrame. coords['In'] is the Series of boolean value whether each frame is 'In food zone' or not.
    start, end: look at lininterpol function for further description
    '''
    df = pd.read_csv(filepath, header=[1, 2], index_col=0, skiprows=0)
    coords, start, end = lininterpol(df, ['Body_center'], ll_crit=ll_crit, absolute=absolute)
    coords['In'] = (coords.values[:, 0] > fz_x)

    return coords, start, end

# D-CF (frequency of bout)

def return_bout(filepath, params, ll_crit=0.7, absolute=True, interval=0.5, FPS=30, latency=3.0):
    '''
    filepath: output of DLC; its bodyparts must contain 'Body_center'

    params: (
        x-coord of tray,
        y-coord of tray,
        threshold of the distance btwn mouth and hand,
        threshold of the distance btwn hand and tray
    )
    
    ll_crit, absolute: look at lininterpol function for further description

    interval: nonnegative float. the minimum time interval (s) btwn two contiguous events of the same hand entering the circle of radius params[2] centered at mouth. default 0.5.
        For example, suppose the right hand entered the circle of radius params[2] centered at mouth at 1.0s.
        In case of interval=0.5, we do not record this event until 1.5s, even if it appears to happen according to the DLC coordinates.
        This is because it is unlikely that the same hand enters the circle more than twice in 0.5s; this is rather because the labeled coordinate is inaccurate for a few frames.
        Hence, we exclude those events happening multiple times in some small time interval, so that our calculation is more compatible with reality.
    
    FPS: FPS of the video. default 30

    latency: positive float. the maximum time interval (s) btwn (1) the event of the hand entering the circle of radius params[3] centered at tray
                                                              & (2) the event of the same hand entering the circle of radius params[2] centered at mouth.
             to consider the event (2) as a bout. default 3.0.
        For example, suppose the event (2) happened at 5.0s. In case of latency=3.0, this event (2) is counted as a bout iff (1) happened in 2.0s~5.0s.

    
    Returns:
    coords: DataFrame. coords['Bout'] is the Series of boolean value whether each frame is 'bout' or not.
    start, end: look at lininterpol function for further description
    '''
    coords = pd.read_csv(filepath, header=[1, 2], index_col=0, skiprows=0).reset_index()
    coords, start, end = lininterpol(coords, ['Mouth', 'R_hand', 'L_hand'], ll_crit=ll_crit, absolute=absolute)

    trayx, trayy, dist0, dist1 = params

    coords['Condition 0R'] = np.linalg.norm(coords['Mouth'].values - coords['R_hand'].values, axis=1) < dist0
    coords['Condition 0L'] = np.linalg.norm(coords['Mouth'].values - coords['L_hand'].values, axis=1) < dist0
    coords['Condition 1R'] = np.linalg.norm(coords['R_hand'].values - np.array([[trayx, trayy]]), axis=1) < dist1
    coords['Condition 1L'] = np.linalg.norm(coords['L_hand'].values - np.array([[trayx, trayy]]), axis=1) < dist1

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
