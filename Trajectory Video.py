# Sample code related to Video S5-6

import numpy as np
import pandas as pd
import cv2

dlc_path = 'DLC_OUTPUT_CSV_FILE_PATH'
video_path = 'VIDEO_PATH'
result_path = 'RESULT_VIDEO_PATH'

bps = ['Head_top', 'Body_center', 'Mouth'] # List of bodyparts

timelength = 0.3 # positive number. time length of the trajectory per frame
thickness = 2 # thickness of the trajectory
ll_crit = 0.7 # Likelihood criterion for excluding values with low likelihood

colors_init = np.array([
                        [252, 180, 214],
                        [220, 249, 222],
                        [255, 204, 185]
]) # numpy array of shape (len(bps), 3). Starting color (BGR) of the trajectory

colors_last = np.array([
                        [255, 24, 128],
                        [160, 240, 165],
                        [253, 111, 53]
]) # numpy array of shape (len(bps), 3). Ending color (BGR) of the trajectory



cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))

traj_length = int(timelength * FPS)

colors_array = np.linspace(colors_init, colors_last, num=traj_length, endpoint=True)

out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width, height))

df = pd.read_csv(dlc_path, header=[1, 2], index_col=0, skiprows=0)

for i in df.index[traj_length:]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, img = cap.read()
    if not ret:
        break
    indices = np.arange(i - traj_length, i + 1)
    for bp_index, bp in enumerate(bps):
        if df.loc[indices, (bp, 'likelihood')].min() >= ll_crit:
            for j in range(-traj_length, 0):
                if j == -traj_length:
                    bp_x_prev, bp_y_prev, bp_x_next, bp_y_next = np.around([df.loc[i+j, (bp, 'x')], df.loc[i+j, (bp, 'y')], df.loc[i+j+1 , (bp, 'x')], df.loc[i+j+1 , (bp, 'y')]]).astype(int)
                    img = cv2.line(img, (bp_x_prev, bp_y_prev), (bp_x_next, bp_y_next), tuple(colors_array[j, bp_index]), thickness)
                else:
                    bp_x_prev, bp_y_prev = bp_x_next, bp_y_next
                    bp_x_next, bp_y_next = np.around([df.loc[i+j+1 , (bp, 'x')], df.loc[i+j+1 , (bp, 'y')]]).astype(int)
                    img = cv2.line(img, (bp_x_prev, bp_y_prev), (bp_x_next, bp_y_next), tuple(colors_array[j, bp_index]), thickness)
    out.write(img)

out.release()
