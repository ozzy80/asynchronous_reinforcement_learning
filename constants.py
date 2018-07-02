# Velicina krajnje slike koja predstavlja ulaz u neuronsku mrezu
FRAME_WIDTH_SIZE = 84
FRAME_HEIGHT_SIZE = 84

# Lokacije fajlova
GAME_SAVE_FOLDER_NAME = "Pong"
GAME_NAME = "Pong-v0" 
CHECKPOINT_DIRECTORY = "./checkpoints/" + GAME_SAVE_FOLDER_NAME

# Parametri za testiranje
THREAD_NUMBER = 12
TMAX = 96000000 # Odgovara 60 epoha iz originalnog rada i 24 epohe iz google-ovog rada
GAMMA = 0.99     
TARGET_UPDATE_FREQUENCY = 30000
OPTIMIZER_UPDATE_FREQUENCY = 32
CHECKPOINT_INTERVAL = 626
EPSILON_SCALE_ITERATION = 4000000
LEARNING_RATE = 0.0001
WRITE_TRAIN_DATA_INTO_FILE = True
MODEL_BACKUP_INTERVAL = 133338
SHOW_GAME_TRANING = True

# Parametri za evaluaciju
NUMBER_OF_TEST_EPISODES = 50


