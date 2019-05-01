#
# Pytorch Double Deep Q-learning for Gym's game
# Min Wang <mingewang@gmail.com>
#
import argparse
import gym
import random
import numpy as np
# list-like container with fast appends and pops on either end
from collections import deque

import math
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_module(nn.Module):
    # env.state_size, output_dim = env.action_space.n
    def __init__(self, input_dim, output_dim):
        super(DQN_module, self).__init__()
        self.l1 = nn.Linear(input_dim, 24)
        self.l2 = nn.Linear(24,24)
        self.l3 = nn.Linear(24,output_dim)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DQN:
    def __init__(self, env, load_model):
        self.env     = env
        self.load_model     = load_model 
        # we save weights here
        self.weights_filename = "ddqn_success.model"
       
        # observation space dimension 
        self.state_size  = env.observation_space.shape[0]
        # action space dimension
        self.predict_size = self.env.action_space.n

        # The Experience Replay buffer stores a fixed number of recent memories
        # and as new ones come in, old ones are removed. 
        # When the time comes to train, we simply draw a uniform batch 
        # of random memories from the buffer, and train our network with them.
        self.memory  = deque(maxlen=2000)

        # hyper parameters for DQN
        self.learning_rate = 0.005
        # discount factor for DQN
        self.gamma = 0.99
        # start with 1 but will decay for each action
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 #0.995

        # minimum sample pool 
        self.train_start = 1000

        # we use batch training to add more stability for the learning process
        self.batch_size = 64
        
        # how do we want to load weights for target model from model's weigths
        self.tau = 1 

        # our output/predict is: reward for each action
        # here we only have two actions: left, right
        # the target = model.predict(state) 
        # is: target[0][0]= reward0
        #     target[0][1]= reward1
        # the first index 0 is for index for the data
        # if we feed an array of data to predict
        # it will output an array of predictions for those data
        self.model        = DQN_module( self.state_size, self.predict_size).to(device)
        self.target_model = DQN_module( self.state_size, self.predict_size).to(device)

        self.criterion = torch.nn.MSELoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.load_model: 
          self.model.load_state_dict( torch.load(self.weights_filename) );

    def act(self, state):
        # will be smaller,smaller as we learned
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if random.random() < self.epsilon:
            # return some random action to explore environment
            return torch.tensor( self.env.action_space.sample() )
        # choose an action which has the max reward based on our model prediction
        return torch.argmax(self.model(state)[0])

    def demo_act(self, state):
        # choose an action which has the max reward based on our model prediction
        return torch.argmax(self.model(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    # major Google DeepMind contribution as DQN
    def replay(self):
        # do not have many samples, do not replay
        if len(self.memory) < self.train_start:
            return

        # enter train mode
        self.model.train()
        self.target_model.eval()

        # we use batch training, initial buffer here
        trainning_input = torch.zeros((self.batch_size, self.state_size))                  
        trainning_target = torch.zeros((self.batch_size, self.predict_size ))   
        trainning_pred = torch.zeros((self.batch_size, self.predict_size ))   

        # Experience Replay, 
        # choose randome samples from our storage
        samples = random.sample(self.memory, self.batch_size)

        for idx, sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            # our model predicts what reward on those actions
            # based on current state
            # the target should be: 
            # target[0][action0] -> reward0
            # target[0][action1] -> reward1
            # ...
            my_predict = self.model(state)
            # store my_predict as model's predict
            trainning_pred[idx] = my_predict

            my_next_predict = self.model(new_state)
            # target model prediction for next new_state
            target = self.target_model(new_state)

            if done:
                my_predict[0][action] = reward
            else:
                # key point of ddqn
                # choose action use model's prediction
                a =  torch.argmax( my_next_predict )
                # but use reward from target model using that action
                Q_future = target[0][a] 
                # update predict training data
                my_predict[0][action] = reward + Q_future * self.gamma

            # store our training data: state->predict
            trainning_input[idx] = state
            # assume to be our ground truth
            trainning_target[idx] = my_predict

        # use NN to do the batch_size training 
        loss = self.criterion( trainning_pred,  trainning_target )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # train target network
    # here we just load weights from model
    def target_train(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save( self.model.state_dict(), self.weights_filename )

def train():
    # load the game
    env     = gym.make("CartPole-v0")
   
    # how many trials we want to play the game 
    trials  = 10000
    # how deep we want
    max_trial_step = 500
    # CartPole-v0 defines "solving" as getting average reward of 195.0 
    # over 100 consecutive trials.
    # to speed up, we check avg of last 10 scores 
    scores  = deque(maxlen=10)
    target_score = 195 
    
    # if game finish earlier, we give a penaty score
    penaty_score = 100 
    max_score = 200 

    training_ok = False

    dqn_agent = DQN(env=env, load_model=False)

    # how many times we want to play
    for trial in range(trials):
        # reset the env for each episode play
        cur_state = torch.from_numpy( env.reset().reshape(1, dqn_agent.state_size) ).float()
        score = 0
        # for each play, how deep/how many steps do we want to play
        for step in range(max_trial_step):
            # render game
            # env.render()
            # action based on our model's prediction
            #import pdb; pdb.set_trace()
            action = dqn_agent.act(cur_state)
            #print( cur_state, action)

            # use gym simulator to go the next step, env take numpy instead of tensor
            new_state, reward, done, _ = env.step( action.numpy() )
            #print( done, new_state, reward, action)

            new_state = torch.from_numpy( new_state.reshape(1,dqn_agent.state_size) ).float()

            # if an action make the episode end, then gives penalty
            # we want this as long as possible
            reward = reward if not done or score == max_score else -penaty_score

            # store play steps 
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            # try to train in every step
            dqn_agent.replay()

            # in this case, score is just  + 1
            score +=  reward
            cur_state = new_state

            if done:
                # iterates target model
                dqn_agent.target_train() 
                # revert back to see avg score
                score = score if score == max_score else score + penaty_score 
                break

        scores.append( score ) 
        avg_score = np.mean(scores)
        print("Episode {}# Score: {}, avg score: {}, ".format(trial, score, avg_score))

        if avg_score > target_score:
          dqn_agent.save_model()
          training_ok = True 
          break

    if training_ok:
      print(" avg score above target score , training OK!")
    else:
      print("Failed to train the model after trials:", trials)

def demo():
    env     = gym.make("CartPole-v0")

    trials  = 20
    max_trial_step = 500

    dqn_agent = DQN(env=env, load_model=True)

    # how many times we want to play
    for trial in range(trials):
        cur_state = torch.from_numpy( env.reset().reshape(1, dqn_agent.state_size) ).float()
        score = 0
        # for each play, how deep/how many steps do we want to play
        for step in range(max_trial_step):
            # render 
            env.render()
            # based on our model's predction, we choose max reward one
            action = dqn_agent.demo_act(cur_state)
            # use simulator to go the next step
            new_state, reward, done, _ = env.step(action.numpy())
            new_state = torch.from_numpy( new_state.reshape(1,dqn_agent.state_size) ).float()
            # in this case, score is just  + 1
            score +=  reward
            cur_state = new_state

            if done:
                break

        print("Demo mode, Episode {}# Score: {}".format(trial, score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--demo',
          action='store_true',
          help='demo flag' )

    args = parser.parse_args()

    if args.demo:
      demo()

    else:
      train()
