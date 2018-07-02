import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import tensorflow as tf
from keras import backend as K
import threading
from game_wrapper import GameWrapper 
from network import update_conv_network
import numpy as np
import time
import csv

from constants import THREAD_NUMBER
from constants import CHECKPOINT_DIRECTORY
from constants import GAME_SAVE_FOLDER_NAME
from constants import GAME_NAME
from constants import TMAX
from constants import GAMMA
from constants import EPSILON_SCALE_ITERATION
from constants import TARGET_UPDATE_FREQUENCY
from constants import CHECKPOINT_INTERVAL
from constants import OPTIMIZER_UPDATE_FREQUENCY
from constants import MODEL_BACKUP_INTERVAL
from constants import WRITE_TRAIN_DATA_INTO_FILE
from constants import SHOW_GAME_TRANING


save_model_folder = CHECKPOINT_DIRECTORY + "/" + GAME_SAVE_FOLDER_NAME
csv_file_directory = CHECKPOINT_DIRECTORY + "/thread_data/thread_info_" + str(THREAD_NUMBER) + "_thread.csv"
csv_file_directory_time = CHECKPOINT_DIRECTORY + "/thread_data/thread_info_" + str(THREAD_NUMBER) + "_thread_time.csv"
thread_directory = CHECKPOINT_DIRECTORY +  '/thread_data' 


def choice_epsilon_limit():
    choice = np.random.rand()
    if choice < 0.4:
        return 0.1
    elif choice < 0.7:
        return 0.01
    else:
        return 0.5   


def choose_next_action(action_size, epsilon, q_values):
    action = np.zeros(action_size)
    if np.random.rand() <= epsilon:
        index = np.random.randint(action_size)
        action[index] = 1
    else:
        index = np.argmax(q_values)
        action[index] = 1
  
    return action, index


def update_epsilon(epsilon, epsilon_limit):
    epsilon_reduce = 0
    if epsilon > epsilon_limit:
        epsilon_reduce = (1 - epsilon_limit) / EPSILON_SCALE_ITERATION
    
    return epsilon-epsilon_reduce 


# Cuvanje kopije modela radi kasnijeg istrazivanja
def backup_save_checkpoints(current_frame):
    backup_file_directory = CHECKPOINT_DIRECTORY + '/backup-' + str(current_frame)
    if not os.path.exists(backup_file_directory):
        os.makedirs(backup_file_directory)

    checkpoint_files = os.listdir(CHECKPOINT_DIRECTORY)
    for file_name in checkpoint_files:
        full_file_name = os.path.join(CHECKPOINT_DIRECTORY, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, backup_file_directory)


def print_time_pretty(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return str(hours) + ":" + str(minutes) + ":" + str(seconds)


def write_in_csv(backup_id, global_time, thread_time, epsilon, reward, q_max):
    with open(csv_file_directory, 'a') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow([backup_id, global_time, thread_time, epsilon, reward, q_max])


def train_thread(thread_id, env, action_size, session, saver, cnn_graph, optimizer_graph):
    # Deljeni theta graf
    state = cnn_graph["state"]
    q_function = cnn_graph["q_function"]

    # Deljeni ciljni graf
    target_state = cnn_graph["state_target"]
    target_q_function = cnn_graph["q_function_target"]
    target_network_update = cnn_graph["target_network_update"]

    # Optimizacioni graf
    action_opt = optimizer_graph["action"]
    reward_opt = optimizer_graph["reward"]
    optimizer_update = optimizer_graph["optimizer_update"]

    # Delkarisanje globalnih katanaca 
    global write_game_info_lock
    global render_lock
    global print_lock

    # Zajednicki brojac frejmova
    global T

    # Inicijalizacija brojaca frejmova pojedinacnih niti
    t = 0

    # Inicijalizacija gradijenta
    return_pack = []
    state_pack = []
    action_pack = []

    # Odredjivanje epsilon-a
    epsilon = 1
    epsilon_limit = choice_epsilon_limit()

    # Obavestenje o pocetku rada niti 
    time.sleep(thread_id*np.random.randint(4)) 
    print("Starting thread {} | final epsilon {}".format(thread_id, epsilon_limit))

    # Uzimanje inicijalnog stanja igrice
    game_over = False
    new_state = env.start_game()

    while T < TMAX:
        # Ispis statistike prethodne etape u terminal i po potrebi u fajl
        if game_over:
            with print_lock:
                print("THREAD: {} | GLOBAL_TIME {} | THREAD TIME {} | EPSILON {:.6f} | REWARD {} | Q_MAX {:.3f}".format(thread_id, T, t, epsilon, reward, mean_q/frames))
                if WRITE_TRAIN_DATA_INTO_FILE:
                    write_in_csv(thread_id, T, t, epsilon, reward, mean_q/frames)


        # Resetovanje igrice                
        env.reset()
        game_over = False

        # Restartovanje brojaca za sledeci prolazak
        mean_q = 0
        frames = 0
        reward = 0

        while not game_over:
            if SHOW_GAME_TRANING:
                with render_lock:
                    env.render()

            # Biranje sledece akcije prema e-pohlepnoj polisi
            q_values = q_function.eval(session = session, feed_dict = {state : [new_state]})
            state_pack.append(new_state)

            action, action_index = choose_next_action(action_size, epsilon, q_values)
            epsilon = update_epsilon(epsilon, epsilon_limit)
            action_pack.append(action)

            # Uzimanje novog stanja i nagrade
            new_state, new_reward, game_over = env.next_step(action_index)

            # Izracunavanje y
            target_q_values = target_q_function.eval(session = session, feed_dict = {target_state : [new_state]})
            if game_over:
                return_pack.append(new_reward)
            else:
                return_pack.append(new_reward + GAMMA*np.max(target_q_values))

            # Povecanje brojaca
            T += 1
            t += 1

            # Izmena target grafa
            if T % TARGET_UPDATE_FREQUENCY == 0:
                session.run(target_network_update)
                print("Target network updated")
    
            # Izmena theta grafa
            if t % OPTIMIZER_UPDATE_FREQUENCY == 0 or game_over:
                if state_pack:
                    session.run(optimizer_update, feed_dict = {reward_opt : return_pack,
                                                            state : state_pack,
                                                            action_opt : action_pack})
                # Ciscenje gradijenata
                return_pack = []
                state_pack = []
                action_pack = []

            # Statisticke promenljive
            mean_q += np.max(q_values)
            frames += 1
            reward += new_reward
    
            # Cuvanje trenutnog stanja mreze i backup stanja mreze radi ispitivanja promena u procesu istrazivanja 
            if t % CHECKPOINT_INTERVAL == 0:
                if not WRITE_TRAIN_DATA_INTO_FILE:
                    saver.save(session, save_model_folder, global_step = t)
                else:
                    with write_game_info_lock: 
                        saver.save(session, save_model_folder, global_step = t)
                        if t % MODEL_BACKUP_INTERVAL == 0 and thread_id == 0:
                            backup_save_checkpoints(t*THREAD_NUMBER)


def train_wrapper(session):  
    # Kreiranje posebne igrice za svaku nit  
    envs = []
    for i in range(THREAD_NUMBER):
        env = GameWrapper(GAME_NAME)
        envs.append(env) 
    
    action_size = len(envs[0].get_actions())
    cnn_graph, optimizer_graph =  update_conv_network(action_size)

    # Inicijalizacija promenjljivih
    session.run(tf.global_variables_initializer())
    session.run(cnn_graph["target_network_update"])
    saver = tf.train.Saver()

    # Pravljenje potrebnih foldera
    if not os.path.exists(CHECKPOINT_DIRECTORY):
        os.makedirs(CHECKPOINT_DIRECTORY)

    if WRITE_TRAIN_DATA_INTO_FILE:
        if not os.path.exists(thread_directory):
            os.makedirs(thread_directory)

        # Pravljenje csv fajla za cuvanje stanja niti tokom izvrsavanja
        with open(csv_file_directory, 'w') as csvfile:
            fieldnames = ["thread_id", "global_time", "thread_time", "epsilon", "reward", "q_max"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    thread_list = []
    for i in range(THREAD_NUMBER):
        thread = threading.Thread(target=train_thread, args=(i, envs[i], action_size, session, saver, cnn_graph, optimizer_graph))
        thread.start()
        thread_list.append(thread)
        
    for thread in thread_list:
        thread.join()
 
   # Kopiranje krajnjeg modela u backup
    backup_save_checkpoints(TMAX)

    for env in envs:
        env.close() 
    

# Inicijalizacija grafa, globalnih podataka i pozivanje omotaca niti
g = tf.Graph()
T = 0
write_game_info_lock = threading.Lock()
render_lock = threading.Lock()
print_lock = threading.Lock()
 
with g.as_default(), tf.Session(graph=g).as_default() as session:
    K.set_session(session)
    start_time = time.time()
    train_wrapper(session)
    end_time = time.time()

    if WRITE_TRAIN_DATA_INTO_FILE:
        elapsed_time = end_time - start_time
        with open(csv_file_directory_time, 'w') as csvfile:
            fieldnames = ["frame_number", "thread_number", "execution_time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'frame_number': TMAX, 'thread_number': THREAD_NUMBER, 'execution_time': print_time_pretty(elapsed_time)})

