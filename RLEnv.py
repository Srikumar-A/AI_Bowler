import gym
import random
import numpy as np
import pandas as pd
import json
class CricketEnv(gym.Env):
    def __init__(self,dataset,valid_dataset=None):
        super(CricketEnv,self).__init__()

        #storing the dataset -offline learning
        self.dataset=dataset
        self.valid_dataset=valid_dataset
        self.n_samples=len(self.dataset)
        self.current_idx=0
        self.test_idx=0
        self.done_list=[]
        
        #defining the action space
        self.action_space=gym.spaces.Box(low=np.array([50.0,10.0,60.0]),high=np.array([300.0,24.0,130.0]),dtype=np.float32)
        
    def reset(self):
        self.current_idx=np.random.randint(0,len(self.dataset))
        self.test_idx=np.random.randint(0,len(self.dataset))
        state=self.get_state()
        self.done_list=[]
        
        return state

    
    def step(self,action,validation=False,n_valid=6):
        done=False
        reward=0
        next_state=self.get_state()
        if not validation:
            self.current_idx=self.next_instance(self.dataset,action,self.done_list)
            self.done_list+=[self.current_idx]
        
            reward=self.dataset.iloc[self.current_idx].reward
            
        
            next_state=self.get_state()
            if len(self.done_list)>=len(self.dataset)/2:
                done=True
            else:
                next_state=self.get_state()
            #print(self.current_idx)

        #validation environment step function        
        else:
            reward=self.dataset.iloc[self.test_idx].reward
            if self.valid_dataset is not None:
                self.test_idx=self.next_instance(self.valid_dataset,action,self.done_list)
                reward=self.valid_dataset.iloc[self.test_idx].reward
            else:
                self.test_idx=self.next_instance(self.dataset,action,self.done_list)
                reward=self.dataset.iloc[self.test_idx].reward
            
            self.done_list+=[self.test_idx]
            if len(self.done_list)>=n_valid:
                done=True
            else:
                next_state=self.get_state()
            #print(self.test_idx)
        info={}
        
        return next_state,reward,done,info
        
    def get_state(self):
        instance=self.dataset.iloc[self.current_idx]
        return instance.state
    #get a 1-NN algorithm to get the closest action to whatever the agent has got
    def next_instance(self,dataset,action,done_list):
        """
        Takes in 3 parameters, 
        1. dataset: pandas dataframe that has line,length and velocity in their columns
        2. action: an array of length in this case.
        3. done_list: a list of indices that the env has explored.
        """
        min_distance=float('inf')
        min_idx=0
        for i in range(len(dataset)):
            if i not in done_list:
                instance=dataset.iloc[i]
                distance=np.sqrt((action[0]-instance.line)**2+(action[1]-instance.length)**2+(action[2]-instance.velocity)**2)
                if distance<min_distance:
                    min_idx=i
                    min_distance=distance
                
        return min_idx    