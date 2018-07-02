import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

from constants import FRAME_WIDTH_SIZE
from constants import FRAME_HEIGHT_SIZE
from constants import LEARNING_RATE


def create_conv_network(scope_name, action_size):
    with tf.name_scope(scope_name):
        current_state = tf.placeholder('float', shape=[None, FRAME_WIDTH_SIZE, FRAME_HEIGHT_SIZE, 4])   
        input_values = Input(shape=(FRAME_WIDTH_SIZE, FRAME_HEIGHT_SIZE, 4))

        first_hidden_layer = Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), padding='same', activation='relu')(input_values)
        second_hidden_layer = Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu')(first_hidden_layer)

        conv_flattened = Flatten()(second_hidden_layer)
        fully_connected = Dense(units=256, activation='relu')(conv_flattened)

        action_values = Dense(units=action_size)(fully_connected) #activation='linear'
        
        model = Model(inputs=input_values, outputs=action_values)
    
    return current_state, model


def update_conv_network(action_size):
    # Deljena mreza
    state, model = create_conv_network(scope_name="shared-conv-network", action_size=action_size)
    network_weights = model.trainable_weights
    q_function = model(state)

    # Ciljna mreza
    state_target, model_target = create_conv_network(scope_name="target-conv-network", action_size=action_size)
    network_weights_target = model_target.trainable_weights
    q_function_target = model_target(state_target)

    # Izmena ciljne mreze uz pomoc podataka iz deljene
    target_network_update = []
    for i in range(len(network_weights_target)):
        tmp = network_weights_target[i].assign(network_weights[i])
        target_network_update.append(tmp)
    

    network_pack = {"state" : state, 
                    "q_function" : q_function,
                    "state_target" : state_target, 
                    "q_function_target" : q_function_target,
                    "target_network_update" : target_network_update
                    }

    optimizer_pack = create_optimizer(action_size, q_function, network_weights)

    return network_pack, optimizer_pack


def create_optimizer(action_size, q_function, network_weights):
    action = tf.placeholder('float', [None, action_size])
    reward = tf.placeholder('float', [None])
    
    q_value = tf.reduce_sum(tf.multiply(q_function, action), reduction_indices=1)
    loss = tf.reduce_mean(tf.square(reward - q_value))
    # Dobro se ponasa za igrice kod kojih akcije nose sirok spektar poena
    #loss = huber_loss(reward, q_value)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=0.90)
    optimizer_update = optimizer.minimize(loss, var_list=network_weights)

    return {"action" : action,
            "reward" : reward,
            "optimizer_update" : optimizer_update
            }


def huber_loss(reward, q_value):
    error = reward - q_value
    if tf.greater(tf.abs(error), 1.0) is not None:
        return tf.abs(error) - 1/2
    return tf.divide(tf.square(error), 2)
