from ActCriNet import ReplayBuffer,ActorNet,CriticNet
import tensorflow as tf
import numpy as np
class AIBowler:
    def __init__(self,input_dims=None,exploration_thresh=0.3,alpha=0.01,beta=0.002,env=None,
                gamma=0.99,n_actions=3,max_size=10000000,tau=0.005,
                batch_size=1,noise_line=15.0,noise_len=3.0,noise_vel=20.0):
        self.gamma=gamma
        self.alpha=alpha
        self.beta=beta
        self.tau=tau
        self.memory=ReplayBuffer(max_size,input_dims,n_actions)
        self.batch_size=batch_size
        self.n_actions=n_actions
        self.noise=noise_line
        self.noise_len=noise_len
        self.noise_vel=noise_vel
        self.max_action=env.action_space.high
        self.min_action=env.action_space.low
        self.exploration_thresh=exploration_thresh

        self.actor=ActorNet(n_actions=n_actions,name='actor')
        self.critic=CriticNet(n_actions=n_actions,name='critic')
        self.target_actor=ActorNet(n_actions=n_actions,name='target_actor')
        self.target_critic=CriticNet(n_actions=n_actions,name='target_critic')

        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self,tau=None):
        if tau is None:
            tau=self.tau

        weights=[]  
        targets=self.target_actor.weights
        #updating target actor network
        print([w.shape for w in self.actor.weights])  # Check weight shapes for actor
        print([w.shape for w in self.target_actor.weights])  # Check weight shapes for target actor

        for i,weight in enumerate(self.actor.weights):
            weights.append(weight*tau+targets[i]*(1-tau)) 
        self.target_actor.set_weights(weights)
        #updating target critic network
        weights=[]
        targets=self.target_critic.weights
        for i,weight in enumerate(self.critic.weights):
            weights.append(weight*tau+targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self,state,action,reward,new_state,done):
        self.memory.store_transition(state,action,reward,new_state,done) 
    def choose_action(self,observation,evaluate=False):
        state= tf.convert_to_tensor([observation],dtype=tf.float32)
        print(state.shape,"Input shape to check whats wrong")
        actions=self.actor(state)
        if not evaluate:
            # using 3 different distributions to sample noise as three elements of actions differ by scale
            actions+=tf.squeeze([tf.random.normal(shape=[1],mean=0.0,stddev=self.noise),
                     tf.random.normal(shape=[1],mean=0.0,stddev=self.noise_len),
                     tf.random.normal(shape=[1],mean=0.0,stddev=self.noise_vel)])
        #clipping the actions to deal with overflow
        actions=tf.squeeze(actions)
        #print("Before clipping: ",actions)
        actions=tf.clip_by_value(actions,self.min_action,self.max_action)
        #print(actions,"some issue here")
        return actions

    def learn(self):
        if self.memory.mem_counter<self.batch_size:
            return
        state,new_state,action,reward,done=self.memory.sample_buffer(self.batch_size)
        states=tf.ragged.constant(state,dtype=tf.float32).to_tensor(default_value=0.0)
        new_states=tf.ragged.constant(new_state,dtype=tf.float32).to_tensor(default_value=0.0)
        actions=tf.convert_to_tensor(action,dtype=tf.float32)
        rewards=tf.convert_to_tensor(reward,dtype=tf.float32)
        dones=tf.convert_to_tensor(done,dtype=tf.float32)
        

        
        #critic update rule
        with tf.GradientTape() as tape:
            target_actions=self.target_actor(new_states)
            critic_value_=self.target_critic(new_states,target_actions)
            critic_value=self.critic(states,actions)
            target=rewards+self.gamma*critic_value_*(1-dones)
            critic_loss=tf.keras.losses.MSE(target,critic_value)
            
        critic_network_gradient=tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient,self.critic.trainable_variables))


        #actor update rule
        with tf.GradientTape() as tape:
            new_policy_actions=self.actor(states)
            actor_loss=-self.critic(states,new_policy_actions)  #- for gradient ascent
            actor_loss=tf.math.reduce_mean(actor_loss)

        actor_network_gradient=tape.gradient(actor_loss,self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient,self.actor.trainable_variables))

        self.update_network_parameters()

    def save_models(self):
        '''
        Save the actor and critic networks that includes target networks too.
        '''
        print("-------- Saving model weights -----------")
        self.actor.save_weights(self.actor.chkpt_file)
        self.critic.save_weights(self.critic.chkpt_file)
        self.target_actor.save_weights(self.target_actor.chkpt_file)
        self.target_critic.save_weights(self.target_critic.chkpt_file)

    def load_models(self):
        '''
        Loading the model weights from checkpoint file.
        '''
        print("------- Loading saved models ----------")
        self.actor.load_weights(self.actor.chkpt_file)
        self.critic.load_weights(self.critic.chkpt_file)
        self.target_actor.load_weights(self.target_actor.chkpt_file)
        self.target_critic.load_weights(self.target_critic.chkpt_file)