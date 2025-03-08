import tensorflow as tf
import os
import tensorflow.keras as keras # type: ignore
from collections import deque
import numpy as np
class CriticNet(keras.Model):
    def __init__(self,n_actions=3,name='critic',chkpt_dir=r'.\tmp\ddpg'):
        super(CriticNet,self).__init__()
        self.model_name=name
        self.chkpt_dir=chkpt_dir
        self.chkpt_file=os.path.join(self.chkpt_dir,self.model_name+'_ddpg.weights.h5')
        #input (n,17,3)
        self.lstm1=keras.layers.LSTM(54,activation='tanh',return_sequences=False)
        self.lstm2=keras.layers.LSTM(16,activation='tanh',return_sequences=False)
        #auxillary input of 3
        self.fc0=keras.layers.Dense(256,activation='relu')
        self.fc1=keras.layers.Dense(64,activation='relu')
        self.fc2=keras.layers.Dense(1,activation='sigmoid')

    def call(self,state,action):
        #looping through timesteps to get the temporal dependencies
        #state=tf.squeeze(state,axis=0)
        for i in range(state.shape[2]):
            feedback=self.lstm1(state[:,:,i,:])
            feedback=tf.expand_dims(feedback,axis=0)
            feedback=self.lstm2(feedback)
        feedback=self.fc0(tf.keras.layers.Flatten()(feedback))
        if len(action.shape)<2:
            action=tf.expand_dims(action,axis=0)   
        #feedback=tf.reshape(feedback,(feedback.shape[0],-1))
        action_value=self.fc1(tf.concat([feedback,action],axis=-1))
        action_value=self.fc2(action_value)
        
        return action_value

class ActorNet(keras.Model):
    def __init__(self,n_actions=3,name='actor',chkpt_dir=r'.\tmp\ddpg'):
        super(ActorNet,self).__init__()
        self.model_name=name
        self.chkpt_dir=chkpt_dir
        self.chkpt_file=os.path.join(self.chkpt_dir,self.model_name+'_ddpg.weights.h5')
        # input (n,17,3)
        self.lstm1=keras.layers.LSTM(54,activation='tanh',return_sequences=False)
        self.lstm2=keras.layers.LSTM(16,activation='tanh',return_sequences=False)
        self.fc1=keras.layers.Dense(64,activation='relu')
        self.fc2=keras.layers.Dense(n_actions,activation='sigmoid')

    def call(self,state):
        #looping through timesteps to learn temporal dependencies
        #state=tf.squeeze(state,axis=0)
        for i in range(state.shape[2]):
            prob=self.lstm1(state[:,:,i,:])
            prob=tf.expand_dims(prob,axis=0)
            prob=self.lstm2(prob)
        prob=self.fc1(prob)
        prob=self.fc2(prob)
        
        return prob*[300,24,130]
    
class ReplayBuffer:
    def __init__(self,mem_size,input_shape=None,n_action_dims=3):
        self.mem_size=mem_size
        self.mem_counter=0
        #self.input_shape=self.input_shape
        self.n_actions_dims=n_action_dims
        self.state_memory=deque([0 for i in range(self.mem_size)])
        self.new_state=deque([0 for i in range(self.mem_size)])
        self.action_memory=deque([0 for i in range(self.mem_size)])
        self.reward_memory=deque([0 for i in range(self.mem_size)])
        self.terminal_memory=deque([0 for i in range(self.mem_size)])

    def store_transition(self,state,action,reward,new_state,done):
        self.state_memory.extendleft([state])
        self.new_state.extendleft([new_state])
        self.action_memory.extendleft([action])
        self.reward_memory.extendleft([reward])
        self.terminal_memory.extendleft([done])

        overflow=len(self.state_memory)-self.mem_size
        for i in range(overflow):
            self.state_memory.pop()
            self.new_state.pop()
            self.action_memory.pop()
            self.reward_memory.pop()
            self.terminal_memory.pop()
        
        self.mem_counter=(self.mem_counter+1)#%self.mem_size
        
    def sample_buffer(self,batch_size):
        max_mem=min(self.mem_size,self.mem_counter)
        inst=np.random.choice(max_mem,batch_size,replace=False)

        states=[self.state_memory[i] for i in inst]
        new_states=[self.new_state[i] for i in inst]
        actions=[self.action_memory[i] for i in inst]
        rewards=[self.reward_memory[i] for i in inst]
        dones=[self.terminal_memory[i]for i in inst]
        return states,new_states,actions,rewards,dones