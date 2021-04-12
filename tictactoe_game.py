
import numpy as np

class State:
    def __init__(self):
        pass

    def hash(self):
        pass

    def is_end(self):
        pass

    def next_stat(self, i,j,symbol):
        '''
        data[i][j]=symbon 이 되는 새로운 state 를 반환
        :param i:
        :param j:
        :param symbol:
        :return: new_state
        '''
        pass

    def print_state(self):
        pass

def get_all_states_impl(current_state, current_symbol, all_states):
    '''
    all_stats={hash:(state, is_end)}
    :param current_state:
    :param current_symbol:
    :param all_states:
    :return:
    '''
    pass

def get_all_states():
    '''
    all_stats={hash:(state, is_end)}\
    :return:all_states
    '''
    pass

all_states=get_all_states()

class Judger:
    pass