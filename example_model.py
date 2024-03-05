import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.batch_size = 16
        self.model = self._build_model()

    # the actual neural network
    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    # helper function to get softmax probabilities based on action values
    # for example if [action1, action2, action3] = [2, -1, 3], the softmax will be [0.27, 0.01, 0.72]
    def softmax_pred(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    # model's memory - used to update q values (model predictions) during the replay function
    # the memory is populated by appending (state, action, reward, next_state, done) tuples after each simulated hand
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # decrease epsilon
    # take a random action if random num is less than epsilon (model exploration)
    # otherwise take an action based on the model's prediction
    # use the softmax function to get probabilities rather than taking the same action everytime for a given state
    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if (np.random.rand() <= self.epsilon):
            return random.randrange(self.action_size), 0
        
        act_values = self.model.predict(state)
        act_values = self.softmax_pred(act_values)

        rand = np.random.uniform(0, 1)

        if rand < act_values[0][0]:
            return 0, act_values
        else:
            return 1, act_values

    # using the replay memory, update model based on predicted reward vs. target reward
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # target is the reward value in memory for the given action
            target = reward

            # very important
            # if this is not the final state, we need to incorporate the reward for the next state
            # initially we only know the reward after the final state, but this helps to slowly update rewards for intermediate states
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            # get model's prediction
            pred = self.model.predict(state)

            # change pred to target for given action
            # for example if pred = [2.14, -0.54, 1.13] and action = 0 and target = 1.97
            # then the pred will be changed to [1.97, -0.54, 1.13]
            pred[0, action] = target
            pred = np.expand_dims(pred, axis=0)

            # update the model for given state based on new pred value
            self.model.fit(state, pred, epochs=1, verbose=0)


# pseudocode for how a training loop would interact with the agent
# for a given number of episodes
#     reset game env
#     while hand is not complete
#         get action from model for given game state
#         take the given action and update game state
#         store experience in replay memory (state, action, reward, next_state, done)
#     update model based on experiences stored in replay memory once there are enough experiences
