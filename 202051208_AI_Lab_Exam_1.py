#!/usr/bin/env python
# coding: utf-8

# # Name : Yenumula Pavan Gopal Mourya
# # Roll no :202051208
# # Exam : AI_Lab_Exam_1

# In[1]:


import numpy as np
import os
import random

# Thresh hold value
epsilon=0.0001
# Discount Factor
Gamma=0.9
# no.of question
No_of_Questions = 10
# States and corresponding rewards and probabilities assosciated
Rewards = dict([(1, (0.99, 100)), 
                     (2, (0.9, 500)), 
                     (3, (0.8, 1000)), 
                     (4, (0.7, 5000)), 
                     (5, (0.6, 10000)),
                     (6, (0.5, 50000)), 
                     (7, (0.4, 100000)), 
                     (8, (0.3, 500000)), 
                     (9, (0.2, 1000000)), 
                     (10, (0.1, 5000000))])

Trans = np.array([[0.99, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.8, 0.2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.7, 0.3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.6, 0.4, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.4, 0.6, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0.3, 0.7, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.8, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.9],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

# CLass of Markov Decision Process
class MDP():

  # Checking if the state is terminal or not
  def Is_Terminal(self, state):
    return True if state == No_of_Questions else False
  # THere are two possible actions at each state
  def Possible_actions(self,state):
    return [] if self.Is_Terminal(state) else ['STAY', 'QUIT']
  # Just a boolean function
  def State_Intialization(self):
    return 1
  
  def Rounds(self):
    return [i for i in range(1, No_of_Questions+2)]
    
  def Reward_S(self, state, action):
        return [(state, 1., 0.)] if state > No_of_Questions else [(state+1, Rewards[state][0], Rewards[state][1]), (No_of_Questions+1, 1.-Rewards[state][0], 0.)] if action == 'STAY' else [(No_of_Questions+1, 1.0, Rewards[state][1])] if state <= No_of_Questions else [(state, 1., 0.)]


# We will make use of bellman equation
# The solution will converge . We can prove it using fixed point theorem
# As mapping of Bellman equation is of contracting nature

def value_iteration(mdp):
    # At the begining fill the value matrix with zeroes
    values = dict(zip(mdp.Rounds(), [0.0] * len(mdp.Rounds())))

    # We will keep on iterating untill we converge
    while True:
        # Difference
        delta = 0

        # Value updation using Bellman's theory
        for state in mdp.Rounds():
            if mdp.Is_Terminal(state):
                continue

            # Initially intialize the max value with - Inifinity
            max_value = -float('inf')
            for action in mdp.Possible_actions(state):
                value = sum(map(lambda x: x[1] * (x[2] + Gamma * values[x[0]]), mdp.Reward_S(state, action)))
                max_value = max(max_value, value)
            # Value updation when absolute difference of  max_value - values[state] is greater than delta
            delta = max(delta, abs(max_value - values[state]))
            values[state] = max_value

        # Checking for convergence
        if delta < epsilon:
            break

    # Optimal Policy Computation
    policy = {}
    for state in mdp.Rounds():
        if mdp.Is_Terminal(state):
            policy[state] = None
        else:
            best_action = max(mdp.Possible_actions(state),
                              key=lambda action: sum(probability * (reward + Gamma * values[next_state])
                                                     for next_state, probability, reward in mdp.Reward_S(state, action)))
            policy[state] = best_action

    # Return the computed values and policy.
    return values, policy

instance = MDP()

Values ,Policy = value_iteration(instance)

print('Policy generated after Value Iteration:')
print(Policy)
print('Values generated after Value Iteration:')
print(Values)


# In[2]:


def policyIteration(mdp):
    # At the begining fill the value matrix with zeroes
    Value =dict(zip(mdp.Rounds(), [0.0] * len(mdp.Rounds())))

    Policy = {s: mdp.Possible_actions(s)[0] for s in mdp.Rounds() if not mdp.Is_Terminal(s)}
    # Infinite Loop until broken
    while True:
         # Infinite Loop until broken (due to convergence)
        while True:
            delta = 0
            for s in mdp.Rounds():
                if mdp.Is_Terminal(s):
                    continue
                Valueal = 0
                for NextState, Probability, Reward in mdp.Reward_S(s, Policy[s]):
                    Valueal += Probability * (Reward + Gamma * Value[NextState])
                # update Valuealue of state
                delta = max(delta, abs(Valueal - Value[s]))
                Value[s] = Valueal
            # When the delta is not signifcant break the infinite loop
            # epsilon is the thresh hold value
            if delta < epsilon:
                break
        # Improvement of policy
        policy_stable = True
        for s in mdp.Rounds():
            if mdp.Is_Terminal(s):
                continue
            old_action = Policy[s]
            # Initially intializing with minus infinity
            MaxValuealue = -float('inf')
            best_action = None
            for a in mdp.Possible_actions(s):
                Valueal = 0
                # For each state in the MDP, compute the value of each possible action under the updated values.
                for NextState, Probability, Reward in mdp.Reward_S(s, a):
                    Valueal += Probability * (Reward + Gamma * Value[NextState])
                if Valueal > MaxValuealue:
                    MaxValuealue = Valueal
                    best_action = a
            # Policy Updation
            # Update the policy to choose the action that has the highest computed value.
            Policy[s] = best_action
            if old_action != best_action:
                # If the policy has changed, set policy_stable to False.
                policy_stable = False
        # If the policy is stable then exit
        if policy_stable:
            break
    # Returning the tuple of corresponding values
    return Value, Policy

instance = MDP()

Values ,Policy = policyIteration(instance)

print('Policy generated after Policy Iteration:')
print(Policy)
print('Values generated after Policy Iteration:')
print(Values)


# In[3]:


# max_steps: an integer specifying the maximum number of steps to take in each sequence. Defaults to 50.
# pi: a dictionary mapping each state to an action, representing the policy used to generate the sequences.
# num_sequences: an integer specifying the number of sequences to generate. Defaults to 1.
def generateSARS(instance, policy, num_sequences=1, max_steps=50):
    # Generate SARS tuples (state, action, reward, next_state) for a given instance and policy. 
    sequences = []
    #It does this by iterating over the specified number of sequences and generating a sequence for each one. 
    for i in range(num_sequences):
        state = instance.State_Intialization()
        sequence = []
        # For each sequence, it starts with an initial state, and then iteratively selects actions using the policy 
        # until a terminal state is reached or a maximum number of steps is taken. 
        for t in range(max_steps):
            if instance.Is_Terminal(state):
                break
            action = policy[state]
            reward_S = instance.Reward_S(state, action)
            next_Rounds, probabilities, rewards = [], [], []
            for rs in reward_S:
                next_Rounds.append(rs[0])
                probabilities.append(rs[1])
                rewards.append(rs[2])

            next_state = random.choices(next_Rounds, probabilities)[0]
            reward = rewards[next_Rounds.index(next_state)]
            sequence.append((state, action, reward, next_state))
            state = next_state
        sequences.append(sequence)
    return sequences

# Example usage
instance = MDP()
policy = {s: instance.Possible_actions(s)[0] for s in instance.Rounds() if not instance.Is_Terminal(s)}
sequences = generateSARS(instance, policy, num_sequences=10, max_steps=20)
print('Sequences :')
print()
i=0
for sequence in sequences:
    print('Sequence', i+1,end=' :')
    print()
    print(sequence)
    print()
    i+=1

# At each step, it selects the next state and reward using the reward function of the instance and the probabilities associated with each possible next state. 
# The resulting sequence of SARS tuples is then added to the list of sequences, which is returned at the end of the function.


# In[4]:



class Monte_Carlo(object):

    def __init__(self, No_of_Questions, Rewards):
        No_of_Questions = No_of_Questions
        Rewards = Rewards

    def game(self, policy):
        # initialize variables
        state = self.State_Intialization()
        done = False
        rewards = []
        # play until the game is over
        while not done:
            # Selecting tje particular action
            action = policy(state)
            # Observation of next state and reward after the corresponding action
            next_state, reward, done = self.Reward_S(state, action)[0]
            # Saving the values
            rewards.append(reward)
            # Updation of state
            state = next_state
        # Cummulative reward compuation
        cumulative_rewards = [sum(rewards[i:]) for i in range(len(rewards))]
        # create a list of (state, action, cumulative reward) tuples
        episodes = [(self.State_Intialization(), None, 0)]
        for i in range(len(rewards)):
            episodes.append((i+1, policy(i+1), cumulative_rewards[i]))
        return episodes

    def State_Intialization(self):
        return 1

    def Is_Terminal(self, state):
        return True if state == No_of_Questions+1 else False

    def Possible_actions(self,state):
        return [] if self.Is_Terminal(state) else ['STAY', 'QUIT']

    def Reward_S(self, state, action):
        return [(state, 1., 0.)] if state > No_of_Questions else [(state+1, Rewards[state][0], Rewards[state][1]), (No_of_Questions+1, 1.-Rewards[state][0], 0.)] if action == 'STAY' else [(No_of_Questions+1, 1.0, Rewards[state][1])] if state <= No_of_Questions else [(state, 1., 0.)]


    def Rounds(self):
        return range(1, No_of_Questions+2)


# Random policy generator function
def random_policy(state):
    Possible_actions = ['STAY', 'QUIT']
    return random.choice(Possible_actions)


# In[5]:


values_list = [v for k, v in Rewards.items()]

# Creating an environment
Environment = Monte_Carlo(10, values_list)

# Creating a random instance or game 
Instance = Environment.game(random_policy)
print('Random game being played : ')
print(Instance)
print()

# # Create the transition matrix
no_of_Rounds = No_of_Questions + 1
possible_no_of_actions = 2

Vector_Reward = np.zeros(no_of_Rounds).astype(int)
for num in range(no_of_Rounds):
    if num == no_of_Rounds - 1:
        # Termination of the game
        # so,no reward will be earned
        Vector_Reward[num] = 0 
    else:
        Vector_Reward[num] = values_list[num][1]


print("Printing the transposition matrix : ")
for i in range(no_of_Rounds):
    for j in range(no_of_Rounds):
        print(Trans[i][j], end = "   ")
    print("\n")
    
print("Vectorical representation of Reward: ")
print(Vector_Reward)



# Let's use Linear Algebra to solve the linear system of equations
identity = np.full((no_of_Rounds, no_of_Rounds), 0)
np.fill_diagonal(identity, 1)
V = np.subtract(identity, Gamma * Trans)
b = Vector_Reward
# value function
V_dash = np.linalg.inv(V)@b 

# Print the value function vector
print("\n Value function : \n")
print(V_dash)

