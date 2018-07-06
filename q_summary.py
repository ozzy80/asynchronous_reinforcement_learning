import os
import numpy as np
import csv
import pandas as pd

from constants import GAME_SAVE_FOLDER_NAME
from constants import CHECKPOINT_DIRECTORY
from constants import TMAX


df = pd.read_csv(CHECKPOINT_DIRECTORY + "/thread_data/thread_info_12_thread.csv")

df = df.drop(['thread_id', 'thread_time', 'epsilon', 'reward'], axis=1)
df = df.groupby(pd.cut(df["global_time"], np.arange(0, TMAX+1, 1600000))).mean()
df.index = np.arange(1600000, TMAX+1, 1600000)
data = []
data.insert(0, {'q_max': 0})

df = df.drop(['global_time', 'Unnamed: 0'], axis=1)
df = df.round(3)
df = pd.concat([pd.DataFrame(data), df])

df.index.name = 'Training_epochs'

if not os.path.exists("results/" + GAME_SAVE_FOLDER_NAME):
    os.makedirs("results/" + GAME_SAVE_FOLDER_NAME)

df.to_csv("results/" + GAME_SAVE_FOLDER_NAME + "/q_value_mean.csv", index=True)