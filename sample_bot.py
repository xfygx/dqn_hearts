#coding=UTF-8
from abc import abstractmethod

from websocket import create_connection
import json
import logging
import sys

import numpy as np
import hearts.network as dqn

USE_DQN=True

dqn_values = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
dqn_suits =  ["D", "C", "H", "S"]

action_idx_to_name = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}
action_name_to_idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}

def sort_cards(cards):
    sorted_cards = sorted(cards, key=lambda x: dqn_values.index(x[0]) * 4 + dqn_suits.index(x[1]))
    return sorted_cards

def card_to_one_hot(card):
    """Converts a single card, in the form of (val, suit), to a one-hot vector.
       The first 13 elements correspond to the value, the last 4 to the suit."""
    if (card == []):
        return np.zeros[17]

    val = dqn_values.index(card[0])
    suit = dqn_suits.index(card[1])

    oh = np.zeros([17])
    oh[val] = 1
    oh[suit + 13] = 1

    return oh

def cards_to_one_hot(cards, n_cards=-1):
    """Converts multiple cards to a concatenation of one-hot vectors."""
    if n_cards < 0:
        n_cards = len(cards)

    oh = np.zeros([17 * n_cards])
    for card_idx, card in enumerate(cards):
        oh[card_idx * 17:(card_idx + 1)*17] = card_to_one_hot(card)

    return oh

def actions_to_one_hot(actions):
    oh = np.zeros([13])
    for action in actions:
        oh[action_name_to_idx[action]] = 1

    return oh > 0

def softmax(x, t):
    x_t = x / t
    x_shift = x_t - np.max(x_t)
    e_x = np.exp(x_shift)
    return e_x / np.sum(e_x)

class Log(object):
    def __init__(self,is_debug=True):
        self.is_debug=is_debug
        self.msg=None
        self.logger = logging.getLogger('hearts_logs')
        hdlr = logging.FileHandler('hearts_logs.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)
    def show_message(self,msg):
        if self.is_debug:
            print("msg")
    def save_logs(self,msg):
        self.logger.info(msg)

IS_DEBUG=True
system_log=Log(IS_DEBUG)

class Card:

    # Takes in strings of the format: "As", "Tc", "6d"
    def __init__(self, card_string):
        self.suit_value_dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9}
        self.suit_index_dict = {"S": 0, "C": 1, "H": 2, "D": 3}
        self.val_string = "AKQJT98765432"
        value, self.suit = card_string[0], card_string[1]
        self.value = self.suit_value_dict[value]
        self.suit_index = self.suit_index_dict[self.suit]

    def __str__(self):
        return self.val_string[14 - self.value] + self.suit

    def toString(self):
        return self.val_string[14 - self.value] + self.suit

    def __repr__(self):
        return self.val_string[14 - self.value] + self.suit
    def __eq__(self, other):
        if self is None:
            return other is None
        elif other is None:
            return False
        return self.value == other.value and self.suit == other.suit

    def __hash__(self):
        return hash(self.value.__hash__()+self.suit.__hash__())

class PokerBot(object):

    def __init__(self,player_name):
        self.round_cards_history=[]
        self.pick_his={}
        self.pick_them={}
        self.round_cards = {}
        self.score_cards={}
        self.player_name=player_name
        self.players_current_picked_cards=[]
        self.game_score_cards = {Card("QS"), Card("TC"), Card("2H"), Card("3H"), Card("4H"), Card("5H"), Card("6H"),
                           Card("7H"), Card("8H"), Card("9H"), Card("TH"), Card("JH"), Card("QH"), Card("KH"),
                           Card("AH")}
    #@abstractmethod
    def receive_cards(self,data):
        err_msg = self.__build_err_msg("receive_cards")
        raise NotImplementedError(err_msg)
    def pass_cards(self,data):
        err_msg = self.__build_err_msg("pass_cards")
        raise NotImplementedError(err_msg)
    def pick_card(self,data):
        err_msg = self.__build_err_msg("pick_card")
        raise NotImplementedError(err_msg)
    def expose_my_cards(self,yourcards):
        err_msg = self.__build_err_msg("expose_my_cards")
        raise NotImplementedError(err_msg)
    def expose_cards_end(self,data):
        err_msg = self.__build_err_msg("expose_cards_announcement")
        raise NotImplementedError(err_msg)
    def receive_opponent_cards(self,data):
        err_msg = self.__build_err_msg("receive_opponent_cards")
        raise NotImplementedError(err_msg)
    def round_end(self,data):
        err_msg = self.__build_err_msg("round_end")
        raise NotImplementedError(err_msg)
    def deal_end(self,data):
        err_msg = self.__build_err_msg("deal_end")
        raise NotImplementedError(err_msg)
    def game_over(self,data):
        err_msg = self.__build_err_msg("game_over")
        raise NotImplementedError(err_msg)
    def pick_history(self,data,is_timeout,pick_his):
        err_msg = self.__build_err_msg("pick_history")
        raise NotImplementedError(err_msg)

    def reset_card_his(self):
        self.round_cards_history = []
        self.pick_his={}
        self.pick_them={}

    def get_card_history(self):
        return self.round_cards_history

    def turn_end(self,data):
        turnCard=data['turnCard']
        turnPlayer=data['turnPlayer']
        players=data['players']
        is_timeout=data['serverRandom']
        for player in players:
            player_name=player['playerName']
            if player_name==self.player_name:
                current_cards=player['cards']
                for card in current_cards:
                    self.players_current_picked_cards.append(Card(card))
        self.round_cards[turnPlayer]=Card(turnCard)
        opp_pick={}
        opp_pick[turnPlayer]=Card(turnCard)
        if (self.pick_his.get(turnPlayer))!=None:
            pick_card_list=self.pick_his.get(turnPlayer)
            pick_card_list.append(Card(turnCard))
            self.pick_his[turnPlayer]=pick_card_list

            pick_card_list=self.pick_them.get(turnPlayer)
            pick_card_list.append(turnCard)
            self.pick_them[turnPlayer]=pick_card_list
            
        else:
            pick_card_list = []
            pick_card_list.append(Card(turnCard))
            self.pick_his[turnPlayer] = pick_card_list

            pick_card_list = []
            pick_card_list.append(turnCard)
            self.pick_them[turnPlayer] = pick_card_list

        self.round_cards_history.append(Card(turnCard))
        self.pick_history(data,is_timeout,opp_pick)

    def get_cards(self,data):
        try:
            receive_cards=[]
            players=data['players']
            for player in players:
                if player['playerName']==self.player_name:
                    cards=player['cards']
                    for card in cards:
                        receive_cards.append(Card(card))
                    break
            return receive_cards
        except Exception as e:
            system_log.show_message(e.message)
            return None

    def get_round_scores(self,is_expose_card=False,data=None):
        if data!=None:
            players=data['roundPlayers']
            picked_user = players[0]
            round_card = self.round_cards.get(picked_user)
            score_cards=[]
            for i in range(len(players)):
                card=self.round_cards.get(players[i])
                if card in self.game_score_cards:
                    score_cards.append(card)
                if round_card.suit_index==card.suit_index:
                    if round_card.value<card.value:
                        picked_user = players[i]
                        round_card=card
            if (self.score_cards.get(picked_user)!=None):
                current_score_cards=self.score_cards.get(picked_user)
                score_cards+=current_score_cards
            self.score_cards[picked_user]=score_cards
            self.round_cards = {}

        receive_cards={}
        for key in self.pick_his.keys():
            picked_score_cards=self.score_cards.get(key)
            round_score = 0
            round_heart_score=0
            is_double = False
            if picked_score_cards!=None:
                for card in picked_score_cards:
                    if card in self.game_score_cards:
                        if card == Card("QS"):
                            round_score += -13
                        elif card == Card("TC"):
                            is_double = True
                        else:
                            round_heart_score += -1
                if is_expose_card:
                    round_heart_score*=2
                round_score+=round_heart_score
                if is_double:
                    round_score*=2
            receive_cards[key] = round_score
        return receive_cards

    def get_deal_scores(self, data):
        try:
            self.score_cards = {}
            final_scores  = {}
            initial_cards = {}
            receive_cards = {}
            picked_cards  = {}
            players = data['players']
            for player in players:
                player_name     = player['playerName']
                palyer_score    = player['dealScore']
                player_initial  = player['initialCards']
                player_receive  = player['receivedCards']
                player_picked   = player['pickedCards']

                final_scores[player_name] = palyer_score
                initial_cards[player_name] = player_initial
                receive_cards[player_name]=player_receive
                picked_cards[player_name]=player_picked
            return final_scores, initial_cards,receive_cards,picked_cards
        except Exception as e:
            system_log.show_message(e.message)
            return None

    def get_game_scores(self,data):
        try:
            receive_cards={}
            players=data['players']
            for player in players:
                player_name=player['playerName']
                palyer_score=player['gameScore']
                receive_cards[player_name]=palyer_score
            return receive_cards
        except Exception as e:
            system_log.show_message(e.message)
            return None

class PokerSocket(object):
    ws = ""
    def __init__(self,player_name,player_number,token,connect_url,poker_bot):
        self.player_name=player_name
        self.connect_url=connect_url
        self.player_number=player_number
        self.poker_bot=poker_bot
        self.token=token

    def takeAction(self,action, data):
       if  action=="new_deal":
           self.poker_bot.receive_cards(data)
       elif action=="pass_cards":
           pass_cards=self.poker_bot.pass_cards(data)
           self.ws.send(json.dumps(
                {
                    "eventName": "pass_my_cards",
                    "data": {
                        "dealNumber": data['dealNumber'],
                        "cards": pass_cards
                    }
                }))
       elif action=="receive_opponent_cards":
           self.poker_bot.receive_opponent_cards(data)
       elif action=="expose_cards":
           export_cards = self.poker_bot.expose_my_cards(data)
           if export_cards!=None:
               self.ws.send(json.dumps(
                   {
                       "eventName": "expose_my_cards",
                       "data": {
                           "dealNumber": data['dealNumber'],
                           "cards": export_cards
                       }
                   }))
       elif action=="expose_cards_end":
           self.poker_bot.expose_cards_end(data)
       elif action=="your_turn":
           pick_card = self.poker_bot.pick_card(data)
           message="Send message:{}".format(json.dumps(
                {
                   "eventName": "pick_card",
                   "data": {
                       "dealNumber": data['dealNumber'],
                       "roundNumber": data['roundNumber'],
                       "turnCard": pick_card
                   }
               }))
           system_log.show_message(message)
           system_log.save_logs(message)
           self.ws.send(json.dumps(
               {
                   "eventName": "pick_card",
                   "data": {
                       "dealNumber": data['dealNumber'],
                       "roundNumber": data['roundNumber'],
                       "turnCard": pick_card
                   }
               }))
       elif action=="turn_end":
           self.poker_bot.turn_end(data) # 用来记录对手的 outgoing 牌
       elif action=="round_end":
           self.poker_bot.round_end(data)
       elif action=="deal_end":
           self.poker_bot.deal_end(data)
           self.poker_bot.reset_card_his()
       elif action=="game_end":
           self.poker_bot.game_over(data)
           self.ws.close()
    def doListen(self):
        try:
            self.ws = create_connection(self.connect_url)
            self.ws.send(json.dumps({
                "eventName": "join",
                "data": {
                    "playerNumber":self.player_number,
                    "playerName":self.player_name,
                    "token":self.token
                }
            }))
            while 1:
                result = self.ws.recv()
                msg = json.loads(result)
                event_name = msg["eventName"]
                data = msg["data"]
                system_log.show_message(event_name)
                system_log.save_logs(event_name)
                system_log.show_message(data)
                system_log.save_logs(data)
                self.takeAction(event_name, data)
        except Exception as e:
            system_log.show_message(e)
            system_log.save_logs(e)
            self.doListen()

class LowPlayBot(PokerBot):

    def __init__(self,name):
        super(LowPlayBot,self).__init__(name)
        self.my_hand_cards=[]
        self.expose_card=False
        self.my_pass_card=[]
        self.deal_score = {}

        self.deal_score['pre'] = {0:0, 1:0, 2:0, 3:0}
        self.deal_score['cur'] = {}
        
        self.trainer = dqn.ToepQNetworkTrainer()

    def data_to_vec(self, cards, cadidate_cards, pick_them, stake, deal_score):

        # cards          = ['4S', '2S', 'QH', 'JH', 'QC', '4C', '4D', '2D']
        # cadidate_cards = ['QC', '4C']
        # pick_them       = {'player4': [2C, KS, AD, QS, 7S], 'player1': [7C, AS, 3D, 6S, TS], 'player2': [JC, TH, 7D, KH, AH], 'player3': [AC, 3S, 5D, 5S, JS, 3C]}        

        current_player_hand = sort_cards(cards)
        print("2222")
        print(pick_them)
        table_vecs = []

        for id in ('4', '1', '2', '3'):
            print("for %s" % id)
            if (('player'+id in pick_them) == False):
                print("player is not exist")
                table_vecs += [np.zeros([17 * 13])]
            else:
                print("player exist")
                table_vecs += [cards_to_one_hot(sort_cards(pick_them['player'+id]), 13)]
        print("3333")
        current_player_hand_vec = cards_to_one_hot(current_player_hand, 13)

        single_values_vec  = [stake]

        score = []
        for i in (4, 1, 2, 3):
            score.append(deal_score['cur'][i-1] - deal_score['pre'][i-1])

        single_values_vec += score

        valid_actions = sort_cards(cadidate_cards)

        all_hands = current_player_hand.copy()
        if (13 > len(current_player_hand)):
            for i in range(0, 13-len(current_player_hand)):  # 不足13张，补足
                all_hands.append('00')
        valid_action_vec = np.array([1 if action in valid_actions else 0 for action in all_hands])
        valid_action_indices = [all_hands.index(action) for action in valid_actions] # 就是有效牌在排序后的手牌中 index
        state_vec = np.concatenate([current_player_hand_vec] + table_vecs + [np.array(single_values_vec)] + [valid_action_vec])

        return state_vec, valid_action_indices, valid_action_vec

    def dqn_predict(self,  state_vec, valid_action_indices, valid_actions_oh):
        network = self.trainer.main_net
        session = self.trainer.session

        Q = session.run(network.Q_predict, feed_dict={network.state_input: [state_vec], network.valid_actions: [valid_actions_oh]})[0]
        #print(Q)
        Q_valid = np.array([Q[idx] for idx in valid_action_indices])
        Q_valid_softmax = softmax(Q_valid, 0.01)
        action_idx = valid_action_indices[np.random.choice(np.arange(0, len(Q_valid_softmax)), p=Q_valid_softmax)]

        return action_idx  # 在手牌中的 index

    def receive_cards(self,data):
        self.my_hand_cards=self.get_cards(data)

    # 换掉三张牌
    def pass_cards(self,data):
        cards = data['self']['cards']
        self.my_hand_cards = []
        for card_str in cards:
            card = Card(card_str)
            self.my_hand_cards.append(card)
        pass_cards=[]
        count=0
        for i in range(len(self.my_hand_cards)):
            card=self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
            if card==Card("QS"):
                pass_cards.append(card)
                count+=1
            elif card==Card("TC"):
                pass_cards.append(card)
                count += 1
        for i in range(len(self.my_hand_cards)):
            card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
            if card.suit_index==2:
                pass_cards.append(card)
                count += 1
                if count ==3:
                    break
        if count <3:
            for i in range(len(self.my_hand_cards)):
                card = self.my_hand_cards[len(self.my_hand_cards) - (i + 1)]
                if card not in self.game_score_cards:
                    pass_cards.append(card)
                    count += 1
                    if count ==3:
                        break
        return_values=[]
        for card in pass_cards:
            return_values.append(card.toString())
        message="Pass Cards:{}".format(return_values)
        system_log.show_message(message)
        system_log.save_logs(message)
        self.my_pass_card=return_values
        return return_values

    # 出牌
    def pick_card(self,data):
        cadidate_cards=data['self']['candidateCards']  # 候选的牌

        cards = data['self']['cards']   # 服务器会送来手上还有的牌
        self.my_hand_cards = []
        for card_str in cards:
            card = Card(card_str)
            self.my_hand_cards.append(card)

        for i in range(0,4):
            self.deal_score['cur'][i] = data['players'][i]['dealScore']

        message = "My Cards:{}".format(self.my_hand_cards)
        system_log.show_message(message)
        card_index=0  # 总是打掉候选牌中的第一张
        message = "Pick Card Event Content:{}".format(data)
        system_log.show_message(message)
        message = "Candidate Cards:{}".format(cadidate_cards)
        system_log.show_message(message)
        system_log.save_logs(message)
        message = "Pick Card:{}".format(cadidate_cards[card_index])
        system_log.show_message(message)
        system_log.save_logs(message)

        if (USE_DQN and len(cadidate_cards) > 1):
            print("=============DQN===================")
            print(sort_cards(cards))            # player hands        
            print(sort_cards(cadidate_cards))   # valid_actions
            print(self.pick_them)    # outgoing
            #print(len(cadidate_cards))

            start_pos = int(data['roundPlayers'][0][-1:]) - 1 # player3  <--
            stake     = int(data['roundNumber'])

            print("aaaa")
            # call network to predict
            print(self.deal_score)
            state_vec, valid_action_indices, valid_action_vec = self.data_to_vec(cards, cadidate_cards,self.pick_them, stake, self.deal_score)
            action_idx = self.dqn_predict(state_vec, valid_action_indices, valid_action_vec)
            draw = sort_cards(cards)[action_idx]
            print("action_idx:{0}, cadidate_cards:{1}, draw:{2}".format(action_idx, cadidate_cards, draw))
            print(draw)
            print(draw)
            print("================================")
            return draw

        return cadidate_cards[card_index] 

    def expose_my_cards(self,yourcards):
        expose_card=[]
        for card in self.my_hand_cards:
            if card==Card("AH"):
                expose_card.append(card.toString())
        message = "Expose Cards:{}".format(expose_card)
        system_log.show_message(message)
        system_log.save_logs(message)
        return expose_card

    def expose_cards_end(self,data):
        players = data['players']
        expose_player=None
        expose_card=None
        for player in players:
            try:
                if player['exposedCards']!=[] and len(player['exposedCards'])>0 and player['exposedCards']!=None:
                    expose_player=player['playerName']
                    expose_card=player['exposedCards']
            except Exception as e:
                system_log.show_message(e.message)
                system_log.save_logs(e.message)
        if expose_player!=None and expose_card!=None:
            message="Player:{}, Expose card:{}".format(expose_player,expose_card)
            system_log.show_message(message)
            system_log.save_logs(message)
            self.expose_card=True
        else:
            message="No player expose card!"
            system_log.show_message(message)
            system_log.save_logs(message)
            self.expose_card=False

    # 收到对手给的三张牌
    def receive_opponent_cards(self,data):
        self.my_hand_cards = self.get_cards(data)
        players = data['players']
        for player in players:
            player_name = player['playerName']
            if player_name == self.player_name:
                picked_cards = player['pickedCards']
                receive_cards = player['receivedCards']
                message = "User Name:{}, Picked Cards:{}, Receive Cards:{}".format(player_name, picked_cards,receive_cards)
                system_log.show_message(message)
                system_log.save_logs(message)

    def round_end(self,data):
        try:
            round_scores=self.get_round_scores(self.expose_card, data)
            for key in round_scores.keys():
                message = "Player name:{}, Round score:{}".format(key, round_scores.get(key))
                system_log.show_message(message)
                system_log.save_logs(message)
        except Exception as e:
            system_log.show_message(e.message)

    def deal_end(self,data):
        self.my_hand_cards=[]
        self.expose_card = False
        deal_scores,initial_cards,receive_cards,picked_cards=self.get_deal_scores(data)
        message = "Player name:{}, Pass Cards:{}".format(self.player_name, self.my_pass_card)
        system_log.show_message(message)
        system_log.save_logs(message)
        for key in deal_scores.keys():
            i = int(key[-1]) - 1
            self.deal_score['cur'][i] = deal_scores.get(key)
            message = "Player name:{}, Deal score:{}".format(key,deal_scores.get(key))
            system_log.show_message(message)
            system_log.save_logs(message)
        for key in initial_cards.keys():
            message = "Player name:{}, Initial cards:{}, Receive cards:{}, Picked cards:{}".format(key, initial_cards.get(key),receive_cards.get(key),picked_cards.get(key))
            system_log.show_message(message)
            system_log.save_logs(message)

        for i in range(0,4):
            self.deal_score['pre'][i] = self.deal_score['cur'][i]   # 保存上一局的最后得分

    def game_over(self,data):
        game_scores = self.get_game_scores(data)
        for key in game_scores.keys():
            message = "Player name:{}, Game score:{}".format(key, game_scores.get(key))
            system_log.show_message(message)
            system_log.save_logs(message)

    def pick_history(self,data,is_timeout,pick_his):
        for key in pick_his.keys():
            message = "Player name:{}, Pick card:{}, Is timeout:{}".format(key,pick_his.get(key),is_timeout)
            system_log.show_message(message)
            system_log.save_logs(message)

def main():
    argv_count=len(sys.argv)
    if argv_count>2:
        player_name = sys.argv[1]
        player_number = sys.argv[2]
        token= sys.argv[3]
        connect_url = sys.argv[4]
    else:
        player_name="player4"
        player_number=99
        token="12345678"
        connect_url="ws://localhost:8080/"
    sample_bot=LowPlayBot(player_name)
    myPokerSocket=PokerSocket(player_name,player_number,token,connect_url,sample_bot)
    myPokerSocket.doListen()

if __name__ == "__main__":
    main()