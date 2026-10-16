"""Make the README figures in assets/."""
import csv
import os

import av
import cv2
import h5py
import numpy as np
from PIL import Image

from splits import HUMAN

S = 128
SHORT = {"put_the_bowl_on_top_of_the_cabinet": "bowl on cabinet", "put_the_wine_bottle_on_top_of_the_cabinet": "bottle on cabinet",
         "put_the_bowl_on_the_plate": "bowl on plate", "put_the_bowl_on_the_stove": "bowl on stove",
         "put_the_cream_cheese_in_the_bowl": "cheese in bowl"}


def video(path):
    return [cv2.resize(f.to_ndarray(format="rgb24"), (S, S), interpolation=cv2.INTER_AREA) for f in av.open(path).decode(video=0)]


def label(im, text):
    im = im.copy()
    cv2.putText(im, text, (3, 12), 0, 0.38, (0, 0, 0), 3)
    cv2.putText(im, text, (3, 12), 0, 0.38, (255, 255, 255), 1)
    return im


def human_clip(task, take):
    seg = next(r for r in csv.DictReader(open("data/human/segments.csv")) if r["file"].endswith(f"{task}_{take:02d}.mov"))
    return video(f"data/human/processed/{task}/{task}_{take:02d}.mp4")[int(seg["f_start"]):int(seg["f_end"])]


def tasks_figure():
    rows = []
    for t in HUMAN:
        h = human_clip(t, 1)
        with h5py.File(f"data/demos/libero_goal/{t}_demo.hdf5") as f:
            r = [cv2.resize(x[::-1], (S, S)) for x in f["data/demo_0/obs/agentview_rgb"][:]]  # stored upside down
        pick = lambda v: [v[0], v[len(v) // 2], v[-1]]
        tiles = [label(x, f"human: {SHORT[t]}" if i == 0 else "") for i, x in enumerate(pick(h))]
        tiles += [label(x, "LIBERO demo" if i == 0 else "") for i, x in enumerate(pick(r))]
        rows.append(np.hstack(tiles[:3] + [np.full((S, 6, 3), 255, np.uint8)] + tiles[3:]))
    cv2.imwrite("assets/tasks.jpg", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def gif(run, task, ep, title):
    h = human_clip(task, 1)
    r = video(f"runs/eval_{run}/videos/{task}_{ep}.mp4")
    h = [h[int(i)] for i in np.linspace(0, len(h) - 1, len(r))]  # play both over the same duration
    step = max(2, len(r) // 60)  # about 60 frames keeps the GIF small; plays at real speed (20 Hz)
    frames = [Image.fromarray(np.hstack([label(a, "human video"), label(b, title)])) for a, b in zip(h, r)][::step]
    frames[0].save(f"assets/{run}_{task}.gif", save_all=True, append_images=frames[1:], duration=50 * step, loop=0)


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    tasks_figure()
    gif("d4", "put_the_bowl_on_top_of_the_cabinet", "00_ok", "robot (D4): success")
    gif("d4", "put_the_wine_bottle_on_top_of_the_cabinet", "01_ok", "robot (D4): success")
    gif("d5", "put_the_bowl_on_the_plate", "00_fail", "robot (D5): failure")
    for f in sorted(os.listdir("assets")):
        print(f, os.path.getsize(f"assets/{f}") // 1024, "KB")
