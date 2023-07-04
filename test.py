
#%% import

import numpy as np
from hmm_py import HMM
from hmmpy_by_alkant import hmm as HMM_ak
from hmm_ysh import train

#%% settint

hidden_states_label = {0 : 'low', 1 : 'mid', 2 : 'high'}
A = np.array([
    [0.2, 0.4, 0.25],
    [0.5, 0.5, 0.15],
    [0.3, 0.1, 0.6],
], dtype=np.float64)
B = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.5, 0.15, 0.15, 0.2],
    [0.3, 0.15, 0.25, 0.3],
], dtype=np.float64)
pi = np.array([0.2, 0.65, 0.15], dtype=np.float64)

A_init = np.array([
    [0.4, 0.25, 0.15],
    [0.15, 0.4, 0.5],
    [0.45, 0.35, 0.35],
], dtype=np.float64)
B_init = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
], dtype=np.float64)
pi_init = np.array([0.3, 0.4, 0.3], dtype=np.float64)

#%% oo

n = 20
T = 10
hmm = HMM(A_init, B_init, pi_init)


def random_select(probs):
    r = np.random.random()
    comp = np.cumsum(probs)
    for i in range(comp.shape[0]):
        if r < comp[i]:
            return i
    return comp.shape[0] - 1


def simul():    
    state = random_select(pi)
    obs = random_select(B[state, :])
    obs_seq = [obs]
    state_seq = [state]
    
    for _ in range(19):
        state = random_select(A[:, state])
        obs = random_select(B[state, :])
        obs_seq.append(obs)
        state_seq.append(state)
        
    return obs_seq, state_seq
    
    
#%% run


input_data = []
for _ in range(2000):
    os, ss = simul()
    input_data.append([os, ss])
    
model = train(input_data)



#%% sdfsdf











hmm = HMM(A_init, B_init, pi_init)
for _ in range(10):
    #print(_)
    os, ss = simul()
    obs_seq = np.array(os, dtype=np.int64)   
    try:
        hmm.baum_welch_train(obs_seq)
    except Exception as e:
        print(e)
    
    
print(A)
print(hmm.A)
print(B)
print(hmm.B)

    


















#%% testcode


import hmm_ysh as HMM_ysh


input_data = list()
input_data.append((['hot','cold','hot'],[3,1,3]))
input_data.append((['hot','hot','hot'],[3,3,3]))
input_data.append((['hot','hot','hot'],[3,3,3]))
input_data.append((['hot','hot','hot'],[3,3,3]))
input_data.append((['hot','hot','hot'],[3,3,3]))
input_data.append((['hot','hot','hot'],[3,3,3]))
input_data.append((['hot','hot','hot'],[3,3,3]))
input_data.append((['hot','hot','hot'],[3,3,3]))
input_data.append((['hot','cold','hot'],[2,1,2]))
input_data.append((['hot','cold','hot'],[1,1,3]))
input_data.append((['hot','cold','cold'],[3,2,2]))
input_data.append((['hot','cold','cold'],[2,1,2]))


test = HMM_ysh.train(input_data, smoothing=0.1)
test.forward([3,1,3])[1]


test._backward([3,1,3])[1]


viterbi, seq = test.decoding([3,1,3])

print(seq)



#%% test2

import collections

# data initialize
transition_dict = collections.defaultdict(dict)
transition_key = ['start','hot','cold','end']
transition_arr = [[0.0, 0.8, 0.2, 0.0],[0.0, 0.6, 0.3, 0.1],[0.0, 0.4, 0.5, 0.1],[0.0, 0.0, 0.0, 0.0]]
for i in range(len(transition_key)):
    for j in range(len(transition_key)):
        transition_dict[transition_key[j]][transition_key[i]]=transition_arr[i][j]

emission_dict = collections.defaultdict(dict)
emission_key = ['hot','cold']
emission_arr = [[0.0,0.0],[0.2,0.5],[0.4,0.4],[0.4,0.1]]

observation = [3,1,3]
for i in range(len(emission_arr)):
    for j in range(len(emission_key)):
        emission_dict[i][emission_key[j]]=emission_arr[i][j]



print(transition_dict['hot']['start'],emission_dict[3]['hot'],transition_dict['hot']['hot'],\
      emission_dict[1]['hot'],transition_dict['cold']['hot'],transition_dict['cold']['end'])


def forward(start_keyword:str, transition_dict:collections.defaultdict(dict), emission_key:list, observation_list:list) -> (collections.defaultdict(dict), float):
    # dp[상태][관측값(T)]
    dp = collections.defaultdict(dict)
    print('Forward Algo')
    
    # start로부터 첫번째 dp를 결정짓기 위함 (initialize)
    for i in emission_key:
        dp[i][0] = transition_dict[i][start_keyword]*emission_dict[observation_list[0]][i]


    for observ in range(1,len(observation_list)):
        for to_emi in emission_key:
            dp[to_emi][observ]=0
            for from_emi in emission_key:
                # print(from_emi,to_emi)
                print('전방('+from_emi+'|'+str(observ-1)+') * '+'전이('+to_emi+'|'+from_emi+') * '+'관측('+str(observation_list[observ])+'|'+from_emi+') '+ 'Target : '+'후방('+to_emi+'|'+str(observ)+')')
                dp[to_emi][observ] = (dp[from_emi][observ-1]*transition_dict[to_emi][from_emi]*emission_dict[observation_list[observ]][to_emi]) + dp[to_emi][observ]

    forward_prob = 0
    for emi in emission_key:
        forward_prob = dp[emi][len(observation_list)-1] + forward_prob
    return dp, forward_prob


def backward(start_keyword:str, transition_dict:collections.defaultdict(dict), emission_key:list, observation_list:list) -> (collections.defaultdict(dict), float):
    # dp[상태][관측값(T)]
    dp = collections.defaultdict(dict)
    print('Backward Algo')
    for observ in range(len(observation_list)-1,-1,-1):
        for to_emi in emission_key:
            # END로는 무조건 끝이라는 1가지 경우의 수밖에 없으니, END로의 전이는 무조건 확률 1
            if observ == len(observation_list)-1:
                dp[to_emi][observ] = 1
                continue
            dp[to_emi][observ]=0
            for from_emi in emission_key:
                # print(from_emi,to_emi)
                print('후방('+from_emi+'|'+str(observ+1)+') * '+'전이('+to_emi+'|'+from_emi+') * '+'관측('+str(observation_list[observ+1])+'|'+from_emi+') '+ 'Target : '+'후방('+to_emi+'|'+str(observ)+')')
                dp[to_emi][observ] = (dp[from_emi][observ+1]*transition_dict[from_emi][to_emi]*emission_dict[observation_list[observ+1]][from_emi]) + dp[to_emi][observ]
    
    backward_prob = 0
    for emi in emission_key:
        backward_prob = dp[emi][0]*transition_dict[emi][start_keyword]*emission_dict[observation[0]][emi] + backward_prob

    return dp, backward_prob


import collections
from typing import Tuple

def decoding(start_keyword:str, transition_dict:collections.defaultdict(dict), emission_key:list, observation_list:list) -> Tuple[list, list]:
    # dp[상태][관측값(T)]
    dp = list()
    print('Viterbi Algo')
    
    # start로부터 첫번째 dp를 결정짓기 위함 (initialize)
    tmp = list()
    for i in emission_key:
        dp.append([transition_dict[i][start_keyword]*emission_dict[observation_list[0]][i]])

    for observ in range(1,len(observation_list)):
        for to_emi in emission_key:
            tmp = list()
            for from_emi in emission_key:
                tmp.append(dp[emission_key.index(from_emi)][observ-1]*transition_dict[to_emi][from_emi]*emission_dict[observation_list[observ]][to_emi])
            dp[emission_key.index(to_emi)].append(max(tmp))

    hidden_states_list = list()    
    for i in zip(*dp):
        hidden_states_list.append((emission_key[i.index(max(i))],max(i)))
    return dp, hidden_states_list


result_for, forward_prob = forward('start',transition_dict,emission_key,observation)
result_back, backward_prob = backward('start',transition_dict,emission_key,observation)
viterbi, hidden_seq = decoding('start',transition_dict,emission_key,observation)



print('3,1,3 개의 순서로 아이스크림을 먹을 확률(forward) :',forward_prob)
print('3,1,3 개의 순서로 아이스크림을 먹을 확률(backward) :',backward_prob)
print('3,1,3 개의 순서로 아이스크림을 먹을때 hidden states SEQ (viterbi decoding) :',hidden_seq)









