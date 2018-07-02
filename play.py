import os
import re
from game_wrapper import GameWrapper 
from network import update_conv_network
import tensorflow as tf
import numpy as np
import keras.backend as K
import csv

from constants import GAME_NAME
from constants import NUMBER_OF_TEST_EPISODES
from constants import CHECKPOINT_DIRECTORY
from constants import SHOW_GAME_TRANING


csv_file_directory = CHECKPOINT_DIRECTORY + "/thread_data/play_average_reward.csv"


# Sortiranje podataka alfa-numericki
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split("([0-9]+)", key) ] 
    return sorted(data, key=alphanum_key)


def write_in_csv(backup_id, avg_score):
    with open(csv_file_directory, 'a') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow([backup_id, avg_score])


def play(session, checkpoint_file):
    # Inicijalizacija igrice
    env = GameWrapper(GAME_NAME)
    
    # Pravljenje grafa
    num_actions = len(env.get_actions())
    cnn_graph,_ = update_conv_network(num_actions)

    # Ucitavanje modela
    saver = tf.train.Saver()
    saver.restore(session, tf.train.latest_checkpoint(checkpoint_file))
    print("Model loaded from {}".format(checkpoint_file))
 
    # Ucitavanje graf 
    state_cnn = cnn_graph["state"]
    q_function_cnn = cnn_graph["q_function"]

    # Odedjivanje srednje vrednosti rezultata meceva
    reward_pack = [] 
    for i in range(NUMBER_OF_TEST_EPISODES):
        new_state = env.start_game()
        game_over = False
        reward = 0
 
        while not game_over:
            if SHOW_GAME_TRANING:
                env.render()

            q_action_values = q_function_cnn.eval(session = session, feed_dict = {state_cnn : [new_state]})
            action_index = np.argmax(q_action_values)
            new_state, new_reward, game_over = env.original_reward_step(action_index)
            reward += new_reward

        reward_pack.append(reward)
        print("Episode {} reward {}".format(i, reward))

    # Ispisivanje statistike meceva
    average_reward = np.sum(reward_pack)
    print("Average reward: {:.3f}".format(average_reward/NUMBER_OF_TEST_EPISODES))
    backup_id = os.path.basename(checkpoint_file).split('-')[1]
    write_in_csv(backup_id, average_reward/NUMBER_OF_TEST_EPISODES)

    env.close()


# Pravljenje csv fajla za statistiku
with open(csv_file_directory, 'w') as csvfile:
    fieldnames = ["backup_id", "average_reward"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Evaluacija backupova
checkpoint_files = sorted_aphanumeric(os.listdir(CHECKPOINT_DIRECTORY))
for file_name in checkpoint_files:
    full_file_name = os.path.join(CHECKPOINT_DIRECTORY, file_name)
    if os.path.isdir(full_file_name) and os.path.basename(full_file_name) != 'thread_data':
        g = tf.Graph()
        with g.as_default(), tf.Session(graph=g).as_default() as session:
            K.set_session(session)
            play(session, full_file_name) 