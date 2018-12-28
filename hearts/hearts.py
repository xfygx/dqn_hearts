#coding=UTF-8
import random
import time

import gym
from gym import spaces, error
from gym.utils import seeding
from numpy import array

from .hearts_core import *

class HeartsEnv(gym.Env):
    def __init__(self, render_delay=None):
        self.n_seed = None
        self.render_delay = render_delay
        self.observation_space = spaces.Tuple([
            # player states
            spaces.Tuple([
                spaces.Discrete(200), # score
                spaces.Tuple([ # hand
                    spaces.MultiDiscrete([13, 4])
                ] * 13),
                spaces.Tuple([ # income
                    spaces.MultiDiscrete([13, 4])
                ] * 52),
            ] * 4),
            # table states
            spaces.Tuple([
                spaces.Discrete(13), # n_round
                spaces.Discrete(4), # start_pos
                spaces.Discrete(4), # cur_pos
                spaces.Discrete(1), # exchanged
                spaces.Discrete(1), # heart_occured
                spaces.Discrete(100), # n_games
                spaces.Discrete(1), # finish_expose
                spaces.Discrete(1), # heart_exposed
                spaces.Tuple([ # board
                    spaces.MultiDiscrete([13, 4])
                ] * 4),
                spaces.Tuple([ # first_draw
                    spaces.MultiDiscrete([13, 4])
                ]),
                spaces.Tuple([ # bank
                    spaces.Tuple([
                        spaces.MultiDiscrete([13, 4])
                    ] * 3),
                ] * 4)
            ]),
        ])

        self.action_space = spaces.Tuple([
            spaces.Discrete(4), # cur_pos
            spaces.Tuple([ # bank(3) draw(1)
                spaces.MultiDiscrete([13, 4])
            ] * 3),
        ])

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        self.n_seed = seed
        random.seed(seed)
        return [seed]

    def render(self, mode='human', close=False):
        self._table.render(mode)
        if self.render_delay:
            time.sleep(self.render_delay)

    def step(self, action):

        draws = []
        cur_pos, card_array = action
        for card in card_array:
            rank = card[0]
            suit = card[1]
            if rank >= 0 and suit >= 0:
                draws.append((rank, suit))

        # (0,1,2,3)/-1
        winner = self._table.step((cur_pos, draws))

        # TODO Maybe can return some debug info
        return winner

    def restart(self):
        self._table.game_start()

    def _pad(self, l, n, v):
        if (not l) or (l is None):
            l = []
        return l + [v] * (n - len(l))

    # return :
    # players 0 ~ 3 
    #   score    
    #   hand cards
    #   outgoing cards
    #   income cards
    #   exchange_out cards
    #   exchange_in cards
    # table
    #   start_pos
    #   current_pos
    #   heart_exposed
    #   round
    #   boards

    def get_current_env(self):
        player_states = []
        for idx, player in enumerate(self._table.players):
            player_features = [
                int(player.score)
            ]
            
            player_hand = player.hand
            # for card in player.hand:
            #     player_hand += card
            # player_hand = self._pad(player_hand, 13, (-1, -1))            # 手牌最多 13 张

            player_outgoing = player.outgoing
            #for card in player.outgoing:
            #    player_outgoing += card
            # player_outgoing = self._pad(player_outgoing, 13, (-1, -1))    # 手牌最多 13 张

            player_income = player.income
            #for card in player.income:
            #    player_income += card
            # player_income = self._pad(player_income, 15, (-1, -1))         # 吃下的牌最多 13 + 1 + 1

            player_exchange_out = player.exchange_out
            #for card in player.exchange_out:
            #    player_exchange_out += card

            player_exchange_in = player.exchange_in
            #for card in player.exchange_in:
            #    player_exchange_in += card

            # Tuple: [int], ([r, s], [r, s], ...), ([r, s], [r, s], ...)
            player_states += [[tuple(player_features), tuple(player_hand), tuple(player_income), tuple(player_outgoing),tuple(player_exchange_out), tuple(player_exchange_in)]]

        # print(player_states)

        table_states = [
            int(self._table.n_round),
            int(self._table.start_pos),
            int(self._table.cur_pos),
            self._table.exchanged,
            self._table.heart_occur,
            int(self._table.n_games),
            self._table.first_draw,
            self._table.board
        ]

        return tuple(player_states), tuple(table_states)

    def _get_current_state(self):
        player_states = []
        for idx, player in enumerate(self._table.players):
            player_features = [
                int(player.score),
            ]
            
            player_hand = []
            for card in player.hand:
                player_hand.append(array(card))
            player_hand = self._pad(player_hand, 13, array((-1, -1)))

            player_income = []
            for card in player.income:
                player_income.append(array(card))
            player_income = self._pad(player_income, 52, array((-1, -1)))

            # Tuple: [int], ([r, s], [r, s], ...), ([r, s], [r, s], ...)
            player_features += [tuple(player_hand), tuple(player_income)]
            player_states += tuple(player_features)

            #print(player_hand)

        table_states = [
            int(self._table.n_round),
            int(self._table.start_pos),
            int(self._table.cur_pos),
            int(self._table.exchanged),
            int(self._table.heart_occur),
            int(self._table.n_games),
            int(self._table.finish_expose),
            int(self._table.heart_exposed),
            int(self._table.orig_player),
        ]

        boards = []
        for card in self._table.board:
            if card:
                boards.append(array(card))
            else:
                boards.append(array((-1, -1)))
        
        #print(boards)

        first_draw = [array(self._table.first_draw)] if self._table.first_draw\
                else [array((-1, -1))]
        #print(first_draw)

        banks = []
        for cards in self._table.bank:
            bank = []
            if cards:
                for card in cards:
                    bank.append(array(card))
            bank = self._pad(bank, 3, array((-1, -1)))
            banks.append(tuple(bank))
        #print(banks)

        table_states += [tuple(boards), tuple(first_draw), tuple(banks)]

        # collect information for DQN
        # current player 
        dqn_states = []
        player_hand = []
        player_outgoing = []
        player_scores = []       
        
        player_hand =self._table.players[self._table.cur_pos].hand

        for idx in range(self._table.cur_pos, self._table.cur_pos + n_players):
            player_outgoing.append(self._table.players[idx % n_players].outgoing) # current player is the first item
            player_scores.append(self._table.players[idx % n_players].score)  # same with above

        dqn_states.append(player_hand)
        dqn_states.append(player_outgoing)
        dqn_states.append(player_scores)
        dqn_states.append(self._table.n_round)

        return (tuple(player_states), tuple(table_states), dqn_states)

    def reset(self):
        self._table = Table(self.n_seed)
        self._table.game_start()
        return self._get_current_state()


