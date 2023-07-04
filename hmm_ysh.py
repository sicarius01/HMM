# -*- coding: utf-8 -*-
# Copyright (c) 2012, Chi-En Wu
# https://github.com/jason2506/PythonHMM/
# 저작권 표시 후 shyu0522 개작

import collections
from math import log
from typing import Tuple

def train(sequences, delta=0.0001, smoothing=0):
    """
    Use the given sequences to train a HMM model.
    This method is an implementation of the `EM algorithm
    <http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_.
    The `delta` argument (which is defaults to 0.0001) specifies that the
    learning algorithm will stop when the difference of the log-likelihood
    between two consecutive iterations is less than delta.
    The `smoothing` argument is used to avoid zero probability,
    see :py:meth:`~hmm.Model.learn`.
    """

    model = _get_init_model(sequences)
    length = len(sequences)
    print('init model==========')
    print('hidden states :', model.states)
    print('observation : ', model.symbols)
    print('start_transition_prob : ', model.start_prob)
    print('transition_prob : ', model.trans_prob)
    print('emission_prob : ', model.emit_prob)
    old_likelihood = 0
    # 위키피디아 EM 알고리즘의 termination 조건을 충족하기 위해서 log-liklihood 를 이용한다.
    for _, symbol_list in sequences:
        old_likelihood += log(model.forward(symbol_list)[1])

    old_likelihood /= length

    while True:
        new_likelihood = 0
        for _, symbol_list in sequences:
            model.learn(symbol_list, smoothing)
            new_likelihood += log(model.forward(symbol_list)[1])
        print(model.start_prob)
        new_likelihood /= length
        # 새로 생성된 모델의 결과와 기존 결과의 우도 차가 사용자가 정의한 threshold(delta)보다 작아지면, stop
        # 즉 MLE를 기반으로 계속 진행하다가, 차이가 별로 없어서 갱신할 수준이 아니면 stop하게 된다.
        # https://www.koreascience.or.kr/article/JAKO200111920779193.pdf
        if abs(new_likelihood - old_likelihood) < delta:
            break

        old_likelihood = new_likelihood
    
    print('trained model==========')
    print('hidden states :', model.states)
    print('observation : ', model.symbols)
    print('start_transition_prob : ', model.start_prob)
    print('transition_prob : ', model.trans_prob)
    print('emission_prob : ', model.emit_prob)

    return model

def _normalize_prob(prob, item_set):
    # 없는 item은 확률 0으로, 나머지는 확률 계산하게된다.
    result = {}
    if prob is None:
        number = len(item_set)
        for item in item_set:
            result[item] = 1.0 / number
    else:
        prob_sum = 0.0
        for item in item_set:
            prob_sum += prob.get(item, 0)

        if prob_sum > 0:
            for item in item_set:
                result[item] = prob.get(item, 0) / prob_sum
        else:
            for item in item_set:
                result[item] = 0

    return result

def _normalize_prob_two_dim(prob, set_from, set_to):
    # 2차원 개념의 dict의 경우 확률 계산
    result = collections.defaultdict(dict)
    if prob is None:
        for item in set_to:
            result[item] = _normalize_prob(None, set_from)
    else:
        for item in set_to:
            result[item] = _normalize_prob(prob.get(item), set_from)

    return result

def _count(item, count):
    if item not in count:
        count[item] = 0
    count[item] += 1

def _count_two_dim(item_from, item_to, count):
    if item_to not in count:
        count[item_to] = {}
    _count(item_from, count[item_to])

def _get_init_model(sequences):
    # 기본적으로 count를 사용해서 초기 확률을 잡는 것으로 model을 초기화한다.
    # 이외에 정규분포 랜덤을 사용한다던지 하는 여러 방법들이 존재하는듯.
    import collections
    symbol_count = collections.defaultdict(dict)
    state_count = collections.defaultdict(dict)
    state_symbol_count = collections.defaultdict(dict)
    state_start_count = collections.defaultdict(dict)
    state_trans_count = collections.defaultdict(dict)

    for state_list, symbol_list in sequences:
        pre_state = None
        for state, symbol in zip(state_list, symbol_list):
            _count(state, state_count)
            _count(symbol, symbol_count)
            _count_two_dim(state, symbol, state_symbol_count)
            if pre_state is None:
                _count(state, state_start_count)
            else:
                _count_two_dim(pre_state, state, state_trans_count)
            pre_state = state
    return Model(state_count.keys(), symbol_count.keys(), state_start_count, state_trans_count, state_symbol_count)

class Model:
    states = None
    symbols = None
    start_prob = None
    trans_prob = None
    emit_prob = None
    gamma = None
    xi = None
    def __init__(self, states, symbols, start_prob=None, trans_prob=None, emit_prob=None, mode=None):
        # 모델의 파라미터를 피클등으로 저장해놓으면 불러서 쓸 수 있게 하려고 조금 수정했다.
        # viterbi에서 dict을 안쓰고, index를 써서 states를 찾기 위해 어쩔 수 없이 list 로 처리하였다.
        # set보다야 성능은 엉망이지만, 우리는 기술 습득을 위한 학습 목적임을 명심하자.
        self.states = list(set(states))
        self.symbols = list(set(symbols))
        if mode is None:
            self.start_prob = _normalize_prob(start_prob, self.states)
            self.trans_prob = _normalize_prob_two_dim(trans_prob, self.states, self.states)
            self.emit_prob = _normalize_prob_two_dim(emit_prob, self.states, self.symbols)
        elif mode.lower()=='load' and start_prob is not None and trans_prob is not None and emit_prob is not None:
            self.start_prob = start_prob
            self.trans_prob = trans_prob
            self.emit_prob = emit_prob
        else:
            raise Exception('Someting is WRONG! when MODEL INIT!')

    def learn(self, sequence, smoothing=0):
        """
        Use the given `sequence` to find the best state transition and
        emission probabilities.
        The optional `smoothing` argument (which is defaults to 0) is the
        smoothing parameter of the
        `additive smoothing <http://en.wikipedia.org/wiki/Additive_smoothing>`_
        to avoid zero probability.
        """
        length = len(sequence)
        # E-Step과 M-Step을 분리하였다.
        def _e_step(sequence):
            # 기존 forward, backward에서는 t를 dict형태로 활용하였으나, 크사이와 람다는 시계열 순이 중요하기에, list로 선언하여 활용하였다.
            alpha = self.forward(sequence)[0]
            beta = self._backward(sequence)[0]
            # print(f'\nalpha : {alpha}\nbeta : {beta}')

            gamma = list()
            for t in range(length):
                all_prob_sum = 0
                gamma.append({})
                for stat_from in self.states:
                    # 각 분자에 해당할 전방*후방을 구하는데, 모든 경우의 수에 대해서 구하는 케이스이므로,
                    gamma[t][stat_from] = alpha[stat_from][t] * beta[stat_from][t]
                    # 연산 할때마다 값을 계속 누적합해주면, 분모까지 한번에 구하는 효과가 된다.
                    all_prob_sum += gamma[t][stat_from]

                if all_prob_sum == 0:
                    # 상태값이 없거나(오류), 상태에서의 확률이 전부 0인경우
                    # 구할 필요가 없는 연산이므로 continue
                    continue
                
                for stat_from in self.states:
                    # t에서 Si가 발생 / t에서 전체 Sn이 발생 => 감마
                    gamma[t][stat_from] = gamma[t][stat_from]/all_prob_sum

            xi = list()
            # xi는 t와 t+1의 전후방 관계를 고려해야 하므로 전체 시퀀스 -1까지 돌리면 전체 다 돔
            for t in range(length-1):
                all_prob_sum = 0
                xi.append(collections.defaultdict(dict))
                for stat_from in self.states:
                    for stat_to in self.states:
                        # print(f't : {t}   stat_from : {stat_from}   stat_to : {stat_to}')
                        # t시점 상태(이전) * t+1시점 상태(이후) * 전이 * t+1 관측
                        # xi[t][stat_to][stat_from] = alpha[stat_from][t]*beta[stat_to][t+1] \
                        #                             * self.trans_prob[stat_to][stat_from] \
                        #                             * self.emit_prob[t+1][stat_to]
                        xi[t][stat_to][stat_from] = alpha[stat_from][t]*beta[stat_to][t+1] 
                        xi[t][stat_to][stat_from] *= self.trans_prob[stat_to][stat_from] 
                        xi[t][stat_to][stat_from] *= self.emit_prob[sequence[t+1]][stat_to]

                        all_prob_sum += xi[t][stat_to][stat_from]
                    
                if all_prob_sum ==0:
                    continue

                for stat_from in self.states:
                    for stat_to in self.states:
                        # t에서 i고 t+1에서 j고, i->j에서 t+1 관측치가 나옴 / t와 t+1 전이에서 가능한 전체 => 크사이
                        xi[t][stat_to][stat_from] = xi[t][stat_to][stat_from]/all_prob_sum
            
            return gamma, xi

        def _m_step(sequence, gamma, xi):
            states_cnt = len(self.states)
            symbols_cnt = len(self.symbols)

            for state_from in self.states:
                # update start probability
                # 단순 접근으로는 gamma[0][state]만 쓰는 것이 맞는데, addictive smoothing 해주려고 일부러 state_cnt를 나누는 것으로 보임.
                # 그래서 나눌 값이 원래는 없는거니까(1) 1을 더해주는 것
                # 파이 업데이트 완료
                self.start_prob[state_from] = \
                    (smoothing + gamma[0][state_from]) / (1 + states_cnt * smoothing)

                # update transition probability
                gamma_sum = 0
                # T-1까지의 감마합
                for t in range(length - 1):
                    gamma_sum += gamma[t][state_from]

                if gamma_sum > 0:
                    # 감마는 크사이로 변경해서 분모로 사용 가능한데, 일단 그냥 감마 쓴다.
                    denominator = gamma_sum + states_cnt * smoothing
                    for state_to in self.states:
                        xi_sum = 0
                        # T-1까지의 크사이 합
                        for t in range(length - 1):
                            xi_sum += xi[t][state_to][state_from]
                        # 전체 합에 대한 나누기 처리로 알파 업데이트 완료
                        self.trans_prob[state_to][state_from] = (smoothing + xi_sum) / denominator
                else:
                    # 0보다 크지 않으면 무조건 0이니까 0이면 계산 할 필요도 없음
                    for state_to in self.states:
                        self.trans_prob[state_to][state_from] = 0

                # update emission probability
                # 베타 업데이트는 T까지 감마 다 더해야 되니, 위에서 더해놨던거에 남은거 1개만 더 더함
                gamma_sum += gamma[length - 1][state_from]
                emit_gamma_sum = {}
                # dict 초기화
                for symbol in self.symbols:
                    emit_gamma_sum[symbol] = 0

                for t in range(length):
                    # 감마를 더하긴 더해야하는데, 분자는 각 관측 값별로 더해서 나눠야함 (st.Ot=Vt 조건)
                    emit_gamma_sum[sequence[t]] += gamma[t][state_from]
                if gamma_sum > 0:
                    denominator = gamma_sum + symbols_cnt * smoothing
                    for symbol in self.symbols:
                        # 전체를 관측치 별로 구분해서 나눈다는게 알파와 가장 큰 차이라면 차이임.
                        # 베타 업데이트 완료
                        self.emit_prob[symbol][state_from] = \
                            (smoothing + emit_gamma_sum[symbol]) / denominator
                else:
                    for symbol in self.symbols:
                        self.emit_prob[symbol][state_from] = 0
        
        self.gamma, self.xi = _e_step(sequence)

        _m_step(sequence, self.gamma, self.xi)

    def __repr__(self):
        return '{name}({_states}, {_symbols}, {_start_prob}, {_trans_prob}, {_emit_prob})' \
            .format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, observation_list:list) -> Tuple[collections.defaultdict, float]:
        # dp[상태][관측값(T)]
        dp = collections.defaultdict(dict)
        # print('Forward Algo')
        
        # start로부터 첫번째 dp를 결정짓기 위함 (initialize)
        for i in self.states:
            dp[i][0] = self.start_prob[i]*self.emit_prob[observation_list[0]][i]

        for observ in range(1,len(observation_list)):
            for to_stat in self.states:
                dp[to_stat][observ]=0
                for from_stat in self.states:
                    # print('전방('+from_stat+'|'+str(observ-1)+') * '+'전이('+to_stat+'|'+from_stat+') * '+'관측('+str(observation_list[observ])+'|'+from_stat+') '+ 'Target : '+'후방('+to_stat+'|'+str(observ)+')')
                    dp[to_stat][observ] = (dp[from_stat][observ-1]*self.trans_prob[to_stat][from_stat]*self.emit_prob[observation_list[observ]][to_stat]) + dp[to_stat][observ]
        
        forward_prob = 0
        for stat in self.states:
            forward_prob = dp[stat][len(observation_list)-1] + forward_prob
        return dp, forward_prob

    def _backward(self, observation_list:list) -> Tuple[collections.defaultdict, float]:
        # dp[상태][관측값(T)]
        dp = collections.defaultdict(dict)
        # print('Backward Algo')
        for observ in range(len(observation_list)-1,-1,-1):
            for to_stat in self.states:
                # END로는 무조건 끝이라는 1가지 경우의 수밖에 없으니, END로의 전이는 무조건 확률 1
                if observ == len(observation_list)-1:
                    dp[to_stat][observ] = 1
                    continue
                dp[to_stat][observ]=0
                for from_stat in self.states:
                    # print('후방('+from_stat+'|'+str(observ+1)+') * '+'전이('+to_stat+'|'+from_stat+') * '+'관측('+str(observation_list[observ+1])+'|'+from_stat+') '+ 'Target : '+'후방('+to_stat+'|'+str(observ)+')')
                    dp[to_stat][observ] = (dp[from_stat][observ+1]*self.trans_prob[from_stat][to_stat]*self.emit_prob[observation_list[observ+1]][from_stat]) + dp[to_stat][observ]

        backward_prob = 0
        for stat in self.states:
            backward_prob = dp[stat][0]*self.start_prob[stat]*self.emit_prob[observation_list[0]][stat] + backward_prob

        return dp, backward_prob
        
    def decoding(self, observation_list:list) -> Tuple[list, list]:
        # dp[상태][관측값(T)]
        # viterbi에서는 2차원 배열로 처리하였다. 내장함수만 써서 max를 효과적으로 처리하기 위함.
        dp = list()
        print('Viterbi Algo')
        
        # start로부터 첫번째 dp를 결정짓기 위함 (initialize)
        for i in self.states:
            dp.append([self.start_prob[i]*self.emit_prob[observation_list[0]][i]])

        # forward를 최대한 그대로 기용하였다. max만 쓰는거로 바꿔줬음
        for observ in range(1,len(observation_list)):
            for to_stat in self.states:
                tmp = list()
                for from_stat in self.states:
                    tmp.append(dp[self.states.index(from_stat)][observ-1]*self.trans_prob[to_stat][from_stat]*self.emit_prob[observation_list[observ]][to_stat])
                dp[self.states.index(to_stat)].append(max(tmp))

        # hidden_states_seq를 구하기 위함.
        # end부터 돌면서 -1로 찾아나가야 FM이긴 한데, 어짜피 dp를 깔끔하게 만들어놓은지라 그냥 순서대로 max index 찾아서 처리하면됨.
        hidden_states_list = list()    
        for i in zip(*dp):
            hidden_states_list.append((self.states[i.index(max(i))],max(i)))
        return dp, hidden_states_list