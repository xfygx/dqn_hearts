import random
import logging
import time
import copy

from numpy import array

from hearts.hearts import HeartsEnv
from hearts.network import ToepQNetworkTrainer
from hearts.hearts_core import RANK_TO_CARD, SUIT_TO_CARD, n_players, n_hands
from hearts.game import sort_cards


logger = logging.getLogger(__name__)

def env_sort_cards(cards):
    sorted_cards = sorted(cards, key=lambda x : x[0] * 4 + x[1])
    return sorted_cards

class BotBase:
    def __init__(self, idx):
        self.idx = idx
        pass

    def declare_action(self, player_obs, table_obs):
        raise NotImplemented()

class RandomBot(BotBase):
    def declare_action(self, player_obs, table_obs):
        action = [self.idx]

        score, (hand,), (income,) = player_obs
        n_round, start_pos, cur_pos, exchanged, hearts_occur, n_game,\
            finish_expose, heart_exposed, orig_player, board, (first_draw,), bank = table_obs

        hand_card = [c for c in hand if c[0] != -1 or c[1] != -1]
        board_card = [c for c in board]

        valid_actions = []

        # if not exchanged and n_game % 4 != 0:
        if not exchanged:
            # 3 cards
            draws = random.sample(hand, 3)
        elif not finish_expose:
            draws = [array([12, 1]), array([-1, -1]), array([-1, -1])]
        else:
            # 1 card
            if self.idx == start_pos and n_round == 0:
                draws = [array([0, 3])]
            else:
                for card in hand_card:
                    if card[1] == first_draw[1]:
                        draws = [card]
                        for c in hand_card:   # 如果找到了同花色的一张牌，就把所有同花色的牌加到valid actions中
                            if c[1] == first_draw[1]:
                                valid_actions.append(c)
                        break
                else:
                    for card in hand_card:
                        (rank, suit) = (card[0], card[1])
                        if n_round == 0:
                            if suit != 1 and not all(card == (10, 0)):
                                draws = [card]
                                for c in hand_card:  # 第一轮，所以，只能出
                                    (rank, suit) = (c[0], c[1])
                                    if suit != 1 and not all( c == (10, 0)):  
                                        valid_actions.append(c)
                                break
                        elif not hearts_occur and suit != 1: # 不是第一轮，但红心没有出现，所有不是红心的都可以出
                            draws = [card]
                            for c in hand_card:
                                (rank, suit) = (c[0], c[1])
                                if suit != 1:
                                    valid_actions.append(c)
                            break
                    else:
                        draws = [random.choice(hand_card)]
                        valid_actions = hand_card

            draws += [array([-1, -1]), array([-1, -1])]

        action.append(tuple(draws))
        return tuple(action), [tuple(n.tolist()) for n in valid_actions]

class SequentialBot(BotBase):
    def declare_action(self, player_obs, table_obs):
        action = [self.idx]

        score, (hand,), (income,) = player_obs
        n_round, start_pos, cur_pos, exchanged, hearts_occur, n_game,\
            finish_expose, heart_exposed, board, (first_draw,), bank = table_obs
        
        hand_card = sorted([c for c in hand if c[0] != -1 or c[1] != -1],\
                key=lambda x: (x[1], x[0]), reverse=True)
        board_card = [c for c in board]

        if not exchanged and n_game % 4 != 0:
            # 3 cards
            draws = random.sample(hand, 3)
        elif not finish_expose:
            draws = [array([12, 1]), array([-1, -1]), array([-1, -1])]
        else:
            # 1 card
            if self.idx == start_pos and n_round == 0:
                draws = [array([0, 3])]
            else:
                for card in hand_card:
                    if card[1] == first_draw[1]:
                        draws = [card]
                        break
                else:
                    for card in hand_card:
                        (rank, suit) = (card[0], card[1])
                        if n_round == 0:
                            if suit != 1 and not all(card == (10, 0)):
                                draws = [card]
                                break
                        elif not hearts_occur and suit != 1:
                            draws = [card]
                            break
                    else:
                        draws = [random.choice(hand_card)]

            draws += [array([-1, -1]), array([-1, -1])]

        action.append(tuple(draws))
        return tuple(action)

class BotProxy:
    def __init__(self, render_delay=None, mode='human'):
        self.bots = [RandomBot(i) for i in range(4)]
        self.env = HeartsEnv(render_delay)
        self.mode = mode
        self.trainer = ToepQNetworkTrainer()

    def add_bot(self, pos, bot):
        self.bots[pos] = bot

    def conv_to_card_predict(self, state, valid_actions):   # 把 rand/suit 转换成 card , pad list
        #print(state)

        # state 只有4个部分，因为这是从obs拿出来的，没有加上动作，有效动作列表
        hand = state[0]
        h = []
        for rank, suit in hand:
            h.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
        state[0] = self.env._pad(sort_cards(h), n_hands, []) # 排序后，补足
        
        outgoing = state[1]
        for i in range(0,n_players):
            o = []
            for rank, suit in outgoing[i]:
                o.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
            state[1][i] = self.env._pad(sort_cards(o), n_hands, [])

        vas = valid_actions  # 有效的动作
        a = []
        for rank, suit in vas:
            a.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
        valid_actions = sort_cards(a)

        #print("round:%d" % state[3])
        #print(state)
        #print(valid_actions)

        return state, valid_actions

    def conv_to_card(self, pre_state, cur_state, valid_actions):   # 把 rand/suit 转换成 card , pad list
        #print(pre_state)
        #print(cur_state)

        hand = pre_state[0]
        h = []
        for rank, suit in hand:
            h.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
        pre_state[0] = self.env._pad(sort_cards(h), n_hands, []) # 排序后，补足
        
        outgoing = pre_state[1]
        for i in range(0,n_players):
            o = []
            for rank, suit in outgoing[i]:
                o.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
            pre_state[1][i] = self.env._pad(sort_cards(o), n_hands, [])

        pre_state[4] = [ (RANK_TO_CARD[rank], SUIT_TO_CARD[suit]) for rank, suit in pre_state[4] ]

        try:
            vas = pre_state[5]  # 有效的动作
            a = []
            for rank, suit in vas:
                a.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
        except Exception as e:
            print(e)
            print(a)
            print(pre_state)
            print("exit01")
            exit(-1)

        pre_state[5] = sort_cards(a)

        hand = cur_state[0]
        h = []
        for rank, suit in hand:
            h.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
        cur_state[0] = self.env._pad(sort_cards(h), n_hands, [])
        
        outgoing = cur_state[1]
        for i in range(0, n_players):
            o = []
            for rank, suit in outgoing[i]:
                o.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
            cur_state[1][i] = self.env._pad(sort_cards(o), n_hands, [])

        cur_state[4] = [ (RANK_TO_CARD[rank], SUIT_TO_CARD[suit]) for rank, suit in cur_state[4] ]

        try:
            vas = cur_state[5]
            a = []
            if (vas != ['all']):
                for rank, suit in vas:
                    a.append((RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
                cur_state[5] = sort_cards(a)
            else:
                cur_state[5] = cur_state[0]
        except Exception as e:
            print(e)
            print(a)
            print(cur_state)
            print("exit02")
            exit(-1)

        valid_actions = cur_state[5]

        #print(pre_state)
        #print(cur_state)

        return pre_state, cur_state, valid_actions


    def train_round(self, pre_state, reward, cur_state, valid_actions):
        
        pre_state, cur_state, valid_actions = self.conv_to_card(pre_state, cur_state, valid_actions)

        return self.trainer.train_round(pre_state, reward, cur_state, valid_actions) # 送入网络

    def cards_to_exchange(self, obs):
        return draws

    def run_a_round(self):
        for i in range(0, 4):
            action, valid_actions = self.bots[cur_pos].declare_action(player_obs, obs[1]) 
            self.env.step(action)

        self.train_round()

    def run_a_deal(self, obs):
        # exchange
        # for 4
        #     step => state
        #     if done , do train
        # return 
        
        obs = self.env.get_current_env()
        for i in range(0, 4):
            self.bots[i].
            draws = card_to_exchange(obs)
            action = , draws
            self.env.step(action)

        for i in range(0, 13):
            run_a_round()

        return


    def run_once(self):
        obs = self.env.reset()   # 游戏开始
        done = False

        episode_idx = 18200000
        target_player = -1   # 2C 起手牌玩家，每个deal期间，只获取该玩家的状态变迁，进行学习
        game_state = 0       # 0: 一局开始, 1: 目标玩家确定

        # rank,suit ===> values, char ===> vector 
        # 0, 3  ===> 2, c  ===>  [1, 0, ....0 / 0, 0, 0, 1]
        pre_state =[]
        cur_state = []
        round_done= False
        while not done:
            if (episode_idx < self.trainer.pretrain_steps+1000): # 使用 bot 来生成 action
                #self.env.render(self.mode)

                n_round, start_pos, cur_pos, exchanged, hearts_occur, n_game,\
                finish_expose, heart_exposed, orig_player, board, (first_draw,), bank = obs[1]

                # 确定了2C牌的玩家
                if (game_state == 0 and target_player == -1 and orig_player != -1 and exchanged == True and finish_expose == True and round_done == False):
                    #print("2c player [%d]" % orig_player)
                    game_state = 1
                    target_player = orig_player # 这时，当前玩家就是目标玩家

                    # 记录当前的状态和动作 到 state
                    pre_state = copy.deepcopy(obs[2])  #  [player hand] [player + others player outgoing] [current all scores] [n_round] / [action] [valid actions]
                    pre_state.append([(0, 3)]) # action
                    pre_state.append([(0, 3)]) # all valid actions
                    #print("====================")
                    #print(pre_state)
                    #print("====================")

                player_obs = tuple([obs[0][i]] for i in range(cur_pos*3, cur_pos*3+3))
                action, valid_actions = self.bots[cur_pos].declare_action(player_obs, obs[1])    # bot 选择自己的准备做出的动作

                #print("start_pos [%d], cur_pos [%d], round_done [%r]" % (start_pos, cur_pos, round_done))
                # 游戏已经开始，一轮结束了，当前玩家等于开始玩家相当于一轮开始，开始玩家等于目标玩家
                if (game_state == 1 and round_done == True and cur_pos == start_pos and start_pos == target_player):
                    # 训练上一轮数据
                    #print("turn orig_player again [%d]" % orig_player)
                    cur_state = copy.deepcopy(obs[2])
                    draw = tuple(action[1][0].tolist())
                    cur_state.append([draw])
                    cur_state.append(valid_actions)

                    reward = 0  # 还没有 winner
                    ep_loss, boltzmann_temp, episode_idx, _ = self.train_round(copy.deepcopy(pre_state), reward, copy.deepcopy(cur_state), valid_actions)
                    pre_state = cur_state
                    cur_state = []

                obs, rew, done, round_done, winner = self.env.step(action) # winner is -1 if deal is not end.

                if (round_done == True and n_round == 12): # n_round 是在step前获得的，所以是12. 最后一个 trick，一个deal打完，winner也出来了。达到1000个 deal，打印一下当前的数据
                    episode_idx += 1
                    if episode_idx % 100 == 0:
                        print("RANDOM Episode {0} L {1} BT {2}".format(episode_idx, ep_loss, boltzmann_temp))

                    #print("The deal is end. winner is [%d]" % winner)

                    # 再训练一轮 带 winner
                    cur_state = copy.deepcopy(obs[2])
                    cur_state.append([(0, 3)]) # 随便加个动作, 只是为了保持格式，不会送入训练
                    cur_state.append(['all'])  # 加上有效动作列表，也只是为了保持格式与 pre_state 一致

                    reward = 0
                    if (winner == target_player):
                        reward = 1
                    else:
                        reward = -1
                    #print("====================")
                    #print(pre_state)
                    #print("====================")
                    ep_loss, boltzmann_temp, episode_idx, ep_buffer = self.train_round(copy.deepcopy(pre_state), reward, copy.deepcopy(cur_state), ['all'])

                    pre_state  = cur_state = []
                    game_state = 0
                    target_player = -1
            else: # train with itself
                #self.env.render(self.mode)

                n_round, start_pos, cur_pos, exchanged, hearts_occur, n_game,\
                finish_expose, heart_exposed, orig_player, board, (first_draw,), bank = obs[1]

                # 确定了2C牌的玩家
                if (game_state == 0 and target_player == -1 and orig_player != -1 and exchanged == True and finish_expose == True and round_done == False):
                    #print("2c player [%d]" % orig_player)
                    game_state = 1
                    target_player = orig_player # 这时，当前玩家就是目标玩家

                    # 记录当前的状态和动作 到 state
                    pre_state = copy.deepcopy(obs[2])  #  [player hand] [player + others player outgoing] [current all scores] [n_round] / [action] [valid actions]
                    pre_state.append([(0, 3)]) # action
                    pre_state.append([(0, 3)]) # all valid actions
                    #print("====================")
                    #print(pre_state)
                    #print("====================")

                player_obs = tuple([obs[0][i]] for i in range(cur_pos*3, cur_pos*3+3))
                action, valid_actions = self.bots[cur_pos].declare_action(player_obs, obs[1])    # bot 选择自己的准备做出的动作
                if (len(valid_actions) > 1 and exchanged == True and finish_expose == True):
                    state, dqn_valid_actions = self.conv_to_card_predict(copy.deepcopy(obs[2]), copy.deepcopy(valid_actions))
                    [action_idx, Q] = self.trainer.action_predict(state, dqn_valid_actions) # state = hand, outgoing*4, round+score*4, valid_action
                    #print("predict:")
                    #print(valid_actions)
                    #print(action_idx)
                    draw = env_sort_cards(obs[2][0])[action_idx]
                    # bot action (1, (array([6, 1]), array([-1, -1]), array([-1, -1]))), predict (1, 1)
                    dqn_action = [action[0]]
                    draws =[array(list(draw))]
                    draws += [array([-1, -1]), array([-1, -1])]
                    dqn_action.append(tuple(draws))
                    dqn_action = tuple(dqn_action)

                    #print("bot action {0}, predict {1}".format(action, dqn_action))
                    action = dqn_action
                else:
                    #print("else action is only one")
                    pass

                #print("start_pos [%d], cur_pos [%d], round_done [%r]" % (start_pos, cur_pos, round_done))
                # 游戏已经开始，一轮结束了，当前玩家等于开始玩家相当于一轮开始，开始玩家等于目标玩家
                if (game_state == 1 and round_done == True and cur_pos == start_pos and start_pos == target_player):
                    # 训练上一轮数据
                    #print("turn orig_player again [%d]" % orig_player)
                    cur_state = copy.deepcopy(obs[2])
                    draw = tuple(action[1][0].tolist())
                    cur_state.append([draw])
                    cur_state.append(valid_actions)

                    reward = 0  # 还没有 winner
                    ep_loss, boltzmann_temp, episode_idx, _ = self.train_round(copy.deepcopy(pre_state), reward, copy.deepcopy(cur_state), valid_actions)
                    pre_state = cur_state
                    cur_state = []

                obs, rew, done, round_done, winner = self.env.step(action) # winner is -1 if deal is not end.

                if (round_done == True and n_round == 12): # n_round 是在step前获得的，所以是12. 最后一个 trick，一个deal打完，winner也出来了。达到1000个 deal，打印一下当前的数据
                    episode_idx += 1
                    if episode_idx % 100 == 0:
                        print("SELF-TRAIN Episode {0} L {1} BT {2}".format(episode_idx, ep_loss, boltzmann_temp))

                    #print("The deal is end. winner is [%d]" % winner)

                    # 再训练一轮 带 winner
                    cur_state = copy.deepcopy(obs[2])
                    cur_state.append([(0, 3)]) # 随便加个动作, 只是为了保持格式，不会送入训练
                    cur_state.append(['all'])  # 加上有效动作列表，也只是为了保持格式与 pre_state 一致

                    reward = 0
                    if (winner == target_player):
                        reward = 1
                    else:
                        reward = -1
                    #print("====================")
                    #print(pre_state)
                    #print("====================")
                    ep_loss, boltzmann_temp, episode_idx, ep_buffer = self.train_round(copy.deepcopy(pre_state), reward, copy.deepcopy(cur_state), ['all'])

                    pre_state  = cur_state = []
                    game_state = 0
                    target_player = -1                

        self.env.render(self.mode)
        return obs
