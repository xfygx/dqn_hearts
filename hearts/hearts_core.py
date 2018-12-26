import os
import random
import logging
import time

from unicards import unicard

# 0:s, 1:h, 2:d, 3:c (-1: None)
# 0:2, 1:3, ..., 8:T, 9:J, 10:Q, 11:K, 12:A (-1: None)
S = 0
H = 1
D = 2
C = 3

RANK_TO_CARD = {
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8,
    7: 9,
    8: 10,
    9: 'J',
    10: 'Q',
    11: 'K',
    12: 'A',
}

SUIT_TO_CARD = {
    0: 's',
    1: 'h',
    2: 'd',
    3: 'c',
}

deck = [(rank, suit) for rank in range(13) for suit in range(4)]
n_players = 4
n_hands = 13

logger = logging.getLogger(__name__)

class Player():
    ALPHA = 4

    def __init__(self):
        self.hand = []             # 手上的牌
        self.income = []           # 吃下的牌
        self.outgoing = []         # 打出去的牌
        self.valid_actions = []    # 当前可用动作
        self.score = 0             # 得分
        self.deal_score = 0        
        self.round_score = 0

    def get_rewards(self, exposed_heart=False):
        heart_broken = 0
        spade_q = False
        club_t = False
        for rank, suit in self.income:
            if (rank, suit) == (10, S):  # Q 黑桃 负13分
                spade_q = True
            elif suit == H:
                heart_broken += 1
            elif (rank, suit) == (8, C):  # 10 梅花  双倍
                club_t = True
        
        rewards = 0
        rewards -= heart_broken * 2 if exposed_heart else heart_broken
        rewards -= 13 if spade_q else 0
        rewards *= 2 if club_t else 1

        # shooting moon
        #if heart_broken == 13 and spade_q:
        #    rewards = (-self.ALPHA) * rewards  

        return rewards

    def sort_hand(self):
        pass


class Table():
    def __init__(self, seed=None):
        self.players = [Player() for _ in range(n_players)]
        self.n_games = 0
        self.backup = [None for _ in range(n_players)]
        self.reset()
        if seed:
            random.seed(seed)

    def reset(self):
        self.n_round = 0
        self.start_pos = 0            # round 的起手牌位置
        self.cur_pos = 0              # 当前玩家
        self.bank = [None for _ in range(n_players)]     # 交换的牌??
        self.exchanged = False
        self.heart_occur = False
        self.board = [None for _ in range(n_players)]     # round 打下的牌
        self.first_draw = None                            # round 中的第一张牌
        self.finish_expose = False
        self.heart_exposed = False
        self.game_over = False
        self.orig_player = -1

    def game_start(self, new_deck=None):
        # Reset Game State
        self.reset()
        self.n_games += 1

        if not new_deck:
            global deck
            random.shuffle(deck)        
        else:
            deck = new_deck

        for i, player in enumerate(self.players):            # 发牌
            player.hand = deck[i*n_hands : (i+1)*n_hands]
            player.income = []
            player.outgoing = []
            player.score = 0

        if self.n_games % 4 == 0:
            self._find_hearts_A()

        self.board = [None for _ in range(n_players)]
        self.backup = [None for _ in range(n_players)]

    def _need_exchange(self):
        # if not self.exchanged and self.n_games % 4 != 0: # 为什么每4盘就不换牌？
        if not self.exchanged:
            return True
        return False

    def _match_suit(self, cur_pos, suit):
        if suit == self.first_draw[1]:
            return True
        for _, s in self.players[cur_pos].hand:
            if s == self.first_draw[1]:
                return False
        return True

    def _shoot_moon(self):
        for i, player in enumerate(self.players):
            hearts = 0
            death = False
            for rank, suit in player.income:
                if suit == H:
                    hearts += 1
                elif (rank, suit) == (10, S):
                    death = True

            if death and hearts == 13:
                return i
        return None

    def _find_clubs_2(self):
        for i, player in enumerate(self.players):
            if (0, C) in player.hand:
                self.start_pos = i         
                self.orig_player = i       # 设置开始的位置  这里可以做为 orig_player

        self.cur_pos = self.start_pos

    def _find_hearts_A(self):
        for i, player in enumerate(self.players):
            if (12, H) in player.hand:
                self.start_pos = i

        self.cur_pos = self.start_pos


    def _clear_screen(self):
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

    def render(self, mode='human'):
        # self._clear_screen()
        print('Game %d' % self.n_games)
        print('Round %d' % self.n_round)

        for i, player in enumerate(self.players):
            print('%s Player %d score %d' % ('>' if i == self.cur_pos else ' ',\
                    i, player.score))
            cards = ''
            for rank, suit in sorted(player.hand, key=lambda c: (c[1], c[0])):
                if mode == 'human':
                    cards += (' '+unicard('%s%s' % (RANK_TO_CARD[rank], SUIT_TO_CARD[suit]),\
                            color=True))
                elif mode == 'ansi':
                    cards += (' %s%s' % (RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
            print(' ', cards)
            cards = ''
            for rank, suit in sorted(player.income, key=lambda c: (c[1], c[0])):
                if mode == 'human':
                    cards += (' '+unicard('%s%s' % (RANK_TO_CARD[rank], SUIT_TO_CARD[suit]),\
                            color=True))
                elif mode == 'ansi':
                    cards += (' %s%s' % (RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
            print(' ', cards, '\n')

        board = ''
        for card in self.board:
            if card:
                rank, suit = card
                if mode == 'human':
                    board += (' '+unicard('%s%s' % (RANK_TO_CARD[rank], SUIT_TO_CARD[suit]),\
                            color=True)+' ')
                elif mode == 'ansi':
                    board += (' %s%s' % (RANK_TO_CARD[rank], SUIT_TO_CARD[suit]))
            else:
                board += ' NA'
        print(board, '\n')

    def step(self, actions):
        round_done = False
        winner = -1
        max_score = -9999

        cur_pos, draws = actions
        logger.debug('[step] cur_pos %r', cur_pos)
        if self.game_over:
            return True, True
        
        if cur_pos != self.cur_pos:
            raise TurnError('Not your turn')
        
        for draw in draws:
            if draw not in self.players[cur_pos].hand:
                raise DrawError('Player %r does not have %r' % (cur_pos, draw))

        if self._need_exchange():
            if self.bank[cur_pos]:
                raise FatalError('Already dropped')
            if len(set(draws)) != 3:
                raise DrawLessThanThreeError('Draws less than 3')

            self.bank[cur_pos] = draws
            for draw in draws:
                self.players[cur_pos].hand.remove(draw)

            if None not in self.bank:
                if self.n_games % 4 == 1:   # pass to left
                    for i, player in enumerate(self.players):
                        player.hand += self.bank[i - 1]
                elif self.n_games % 4 == 2: # pass to right
                    for i, player in enumerate(self.players):
                        player.hand += self.bank[(i + 1) % 4]
                elif self.n_games % 4 == 3: # pass to cross
                    for i, player in enumerate(self.players):
                        player.hand += self.bank[(i + 2) % 4]
                else:
                    #raise FatalError('Game# mod 4 == 0')
                    for i, player in enumerate(self.players):
                        player.hand += self.bank[(i) % 4]

                self.exchanged = True
                self._find_hearts_A()
            else:
                self.cur_pos = (self.cur_pos + 1) % n_players
        elif not self.finish_expose:
            if len(draws) > 1:
                raise DrawMoreThanOneError('Draw more than 1 card')

            if len(draws) == 1:
                if draws[0] != (12, H):
                    raise ExposeError('Must expose hearts A')
                self.heart_exposed = True

            self.finish_expose = True
            self._find_clubs_2()     # 找到有 梅花2 的玩家
        else:
            if len(draws) > 1:
                raise DrawMoreThanOneError('Draw more than 1 card')

            draw = draws[0]
            rank, suit = draw
            if self.start_pos == cur_pos:
                if self.n_round == 0 and (0, C) != draw:      # 第一张牌必须是 梅花2, 这里C就是3
                    raise FirstDrawError('The first draw must be (0, 3)')
                if not self.heart_occur and suit == H:
                    for card in self.players[cur_pos].hand:
                        if card[1] != H:
                            raise HeartsError('Cannot draw HEART')
                self.first_draw = draw
            else:
                if not self.first_draw:
                    raise FatalError('You are not the first one')
                if self.n_round == 0 and (draw == (10, S) or suit == H):
                    for card in self.players[cur_pos].hand:
                        if card[1] != H and card != (10, S):
                            raise FirstRoundError('First round cannot break')
                if not self._match_suit(cur_pos, suit):
                    raise RuleError('Suit does not match')

            self.board[cur_pos] = draw
            if suit == H or draw == (10, S):
                self.heart_occur = True
            self.players[cur_pos].hand.remove(draw)
            self.players[cur_pos].outgoing.append(draw)  # 打出一张牌

            if None not in self.board:
                max_rank, first_suit = self.first_draw
                for i, (board_rank, board_suit) in enumerate(self.board):
                    if first_suit == board_suit and board_rank > max_rank:
                        max_rank = board_rank

                self.start_pos = self.board.index((max_rank, first_suit)) # 计算下一轮是哪位玩家
                self.players[self.start_pos].income += self.board   # 收下牌
                self.backup = self.board
                self.board = [None for _ in range(n_players)]
                self.first_draw = None
                round_done = True
                self.n_round += 1   # 这一 trick 打完
                self.cur_pos = self.start_pos

                for player in self.players:
                    score = player.deal_score + player.get_rewards(self.heart_exposed) # 玩家自已算分
                    player.round_score = score - player.score
            else:
                self.cur_pos = (self.cur_pos + 1) % n_players
            
        for player in self.players:
            if self.n_round == 13:
                player.deal_score += player.get_rewards(self.heart_exposed)
                player.score = player.deal_score
            else:
                player.score = player.deal_score + player.get_rewards(self.heart_exposed) # 玩家自已算分

        if self.n_round == 13:
            #if self.n_games == 16:
                # Game Over
            #    return True, round_done

            # 计算 winner
            for i, player in enumerate(self.players):
                if (player.score > max_score):
                    winner = i
                    max_score = player.score
                player.deal_score = player.score = 0 # 清掉积累的分，原程序是会累积的

            round_done = True
            self.game_start()   # 开始新的 deal 

        return False, round_done, winner

    def _step(self, actions):
        self.step(actions)
        self.render()

class TurnError(Exception):
    pass

class DrawMoreThanOneError(Exception):
    pass

class DrawLessThanThreeError(Exception):
    pass

class DrawError(Exception):
    pass

class FatalError(Exception):
    pass

class FirstDrawError(Exception):
    pass

class HeartsError(Exception):
    pass

class RuleError(Exception):
    pass

class FirstRoundError(Exception):
    pass

class ExposeError(Exception):
    pass
