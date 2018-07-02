import gym
import tensorflow as tf
import re
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import Counter, deque

from constants import FRAME_WIDTH_SIZE
from constants import FRAME_HEIGHT_SIZE


class GameWrapper:
    def __init__(self, game):
        self.env = gym.make(game)
        self.historic_frame_pack = deque()

        if re.search("Pong", game):
            self.action_set = [action for action in range(1, 4)]

        elif re.search("Breakout", game):
            self.action_set = [action for action in range(1, 4)]

        elif re.search("Freeway", game):
            self.action_set = [action for action in range(1, 4)]

        elif re.search("Enduro", game):
            self.action_set = [action for action in range(1, 5)]

        else:
            self.action_set = [action for action in range(1, self.env.action_space.n)]


    def start_game(self):
        frame = self.env.reset()
        self.historic_frame_pack = deque()

        frame = rgb2gray(frame)
        output_shape = (FRAME_WIDTH_SIZE, FRAME_HEIGHT_SIZE)
        frame = resize(image=frame, output_shape=output_shape, order=1, mode='constant')
        self.historic_frame_pack.append(frame)
        self.historic_frame_pack.append(frame)
        self.historic_frame_pack.append(frame)
        self.historic_frame_pack.append(frame)

        return np.stack((frame, frame, frame, frame), axis = 2)


    def next_step(self, action_index):
        new_states, reward, game_over = self.original_reward_step(action_index)
        return new_states, np.sign(reward), game_over

    def original_reward_step(self, action_index):
        action = self.action_set[int(action_index)]
        frame, reward, game_over, _ = self.env.step(action)
        frame = rgb2gray(frame)
        output_shape = (FRAME_WIDTH_SIZE, FRAME_HEIGHT_SIZE)
        frame = resize(image=frame, output_shape=output_shape, order=1, mode='constant')

        self.historic_frame_pack.popleft()
        self.historic_frame_pack.append(frame)
        new_states = np.stack((self.historic_frame_pack[0], self.historic_frame_pack[1], self.historic_frame_pack[2], self.historic_frame_pack[3]), axis = 2)

        return new_states, reward, game_over

    def get_actions(self):
        return self.action_set

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def reset(self):
        self.env.reset()