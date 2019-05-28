import cv2
import numpy as np

dir = '../ArenaGameVideos'

H = int(670 * 4)
W = int(1430 * 4)

Y = 4
X = 5

XY = int(X * Y)

h = int(H / Y)
w = int(W / X)

finish = [False] * XY

caps = []


def cap_video(video_i):
    return cv2.VideoCapture('{}/{}.mov'.format(dir, video_i), 0)


for video_i in range(XY):
    caps += [cap_video(video_i)]

out = cv2.VideoWriter('{}/output.mp4'.format(dir),
                      cv2.VideoWriter_fourcc(*'XVID'), caps[0].get(cv2.CAP_PROP_FPS), (W, H))

while True:

    frames = []
    for video_i in range(XY):
        while True:
            ret, frame = caps[video_i].read()
            if ret:
                frames += [frame]
                break
            else:
                caps[video_i] = cap_video(video_i)
                finish[video_i] = True

    video_i = 0
    cat_frame_col = None
    for y in range(Y):
        cat_frame_row = None
        for x in range(X):
            frames[video_i] = cv2.resize(frames[video_i], (w, h))
            if cat_frame_row is None:
                cat_frame_row = frames[video_i]
            else:
                cat_frame_row = np.concatenate(
                    (cat_frame_row, frames[video_i]), 1)

            video_i += 1
        if cat_frame_col is None:
            cat_frame_col = cat_frame_row
        else:
            cat_frame_col = np.concatenate((cat_frame_col, cat_frame_row), 0)

    print(cat_frame_col.shape)
    out.write(cat_frame_col)

    # cv2.imshow('Frame', cat_frame_col)
    # cv2.waitKey(1)

    print(finish)
    if np.all(np.asarray(finish)):
        for cap in caps:
            cap.release()
        out.release()
