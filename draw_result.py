import matplotlib.pyplot as plt
import pandas as pd

from constants import CHECKPOINT_DIRECTORY
from constants import GAME_SAVE_FOLDER_NAME

csv_folder = "results/" + GAME_SAVE_FOLDER_NAME


# Iscrtavanje odnosa q vrednosti
q_value_csv_file = csv_folder + "/q_value_mean.csv"
q_value_paper_csv_file = csv_folder + "/q_value_mean_paper.csv"
df_q = pd.read_csv(q_value_csv_file)
df_q_paper = pd.read_csv(q_value_paper_csv_file)

plt.plot(df_q['q_max'], label='Q-vrednost 12 niti')
plt.plot(df_q_paper['Average_action_value'], label='Q-vrednost originalni rad')
plt.legend()
plt.title("Prosecan Q postignut na Breakout")
plt.ylabel("Prosecna vrednost akcije (Q)")
plt.xlabel("Epohe treniranja")
plt.show()


# Iscrtavanje odnosa poena
reward_8_threads_csv_file = csv_folder + "/reward_mean_8_threads_paper.csv"
reward_12_threads_csv_file = csv_folder + "/play_average_reward.csv"
reward_16_threads_csv_file = csv_folder + "/reward_mean_16_threads_paper.csv"
reward_paper_csv_file = csv_folder + "/reward_mean_paper.csv"
df_reward_8_threads = pd.read_csv(reward_8_threads_csv_file)
df_reward_12_threads = pd.read_csv(reward_12_threads_csv_file)
df_reward_16_threads = pd.read_csv(reward_16_threads_csv_file)
df_reward_paper = pd.read_csv(reward_paper_csv_file)

plt.plot(df_reward_16_threads['average_reward'], c='red', linestyle='--', label='Prosek poena 16 niti')
plt.plot(df_reward_8_threads['average_reward'], c='g', linestyle='--', label='Prosek poena 8 niti')
plt.plot(df_reward_12_threads['average_reward'], c='darkblue', label='Prosek poena 12 niti')
#plt.plot(df_reward_paper['average_reward'], c='gold', label='Prosek poena originalni rad')
plt.legend()
plt.title("Prosecni poeni postignuti na Breakout")
plt.ylabel("Prosecna vrednost poena")
plt.xlabel("Epohe treniranja")
plt.grid()
plt.show()