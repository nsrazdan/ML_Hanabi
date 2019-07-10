# Copy over code from earlier version.
# Load the data if it already exists, otherwise create it

from utils import parse_args
from utils import dir_utils
import gin
from subprocess import call
import pickle
import random
import numpy as np

#constant variables
NUM_ADHOC_GAMES = 10
OBS_VEC_LEN = 658
ACT_VEC_LEN = 20
OBS_ACT_VEC_LEN = OBS_VEC_LEN + ACT_VEC_LEN

@gin.configurable
class Dataset(object):
    @gin.configurable
    def __init__(self, 
            game_type='Hanabi-Full',
            num_players=2,
            num_unique_agents=6,
            num_games=150):

        self.game_type = game_type
        self.num_players = num_players
        self.num_unique_agents = num_unique_agents
        self.num_games = num_games

        self.train_data = {} # gameplay data given to model
        self.validation_data = {} # data not given to model, from same agents as train
        self.test_data = {} # data from agents totally unseen to model
        
    def read(self, raw_data):
        # split up raw_data into train, validation, and test
        #test_agent = random.choice(list(raw_data.keys()))
        
        '''
        for agent in raw_data:
            if agent == test_agent:
                continue
            split_idx = int(0.9 * len(raw_data[agent]))
            self.train_data[agent] = raw_data[agent][:split_idx]
            self.validation_data[agent] = raw_data[agent][split_idx:]
        
        self.test_data[test_agent] = raw_data[test_agent]
        '''
        for agent in raw_data:
            split_idx = int(0.9 * (len(raw_data[agent])-100))
            self.train_data[agent] = raw_data[agent][:split_idx]
            self.validation_data[agent] = raw_data[agent][split_idx:]
            self.test_data[agent] = raw_data[agent][len(raw_data[agent])-100:]

    @gin.configurable
    def generator(self, batch_type='train', shuffle='false',agent='rainbow_agent_1'):
        NUM_ADHOC_GAMES = self.num_games
        if batch_type == 'train':
            data_bank = self.train_data
        elif batch_type == 'validation':
            data_bank = self.validation_data
        elif batch_type == 'test':
            data_bank = self.test_data
            NUM_ADHOC_GAMES = 100
       
        # data_bank: [AgentName][num_games][0 = 
        #         obs_vec, 1 = act_vec][game_step][index into vec]
        #List of all agents. We chose randomely 1 agent
        #agent = random.choice(list(data_bank.keys()))
        #10 ad_hoc games which were played by agent
        adhoc_games = [random.choice(list(data_bank[agent])) 
                for _ in range(NUM_ADHOC_GAMES)]
        #number of round in each game
        game_lengths = [len(game[0]) for game in adhoc_games]
        MAX_GAME_LEN = max(game_lengths)
        
        # adhoc_games: [-->[[obs_act_vec],[obs_act_vec],...]<--game1, 
        #               -->[[obs_act_vec],[obs_act_vec],...]<--game2...]
        adhoc_games = [[adhoc_games[i][0][l] + adhoc_games[i][1][l] 
                       for l in range(game_lengths[i])] 
                       for i in range(NUM_ADHOC_GAMES)]
        # assemble generated agent observations and actions into x and y array
        # NUM_TOTAL_MOVES is the sum of the length of all 10 adhoc_games (total number of 
        # observations/actions throughout all 10 games
        NUM_TOTAL_MOVES = np.sum(game_lengths)
        agent_obs, agent_act = [], []

        # choose from all observations/actions randomly, with all equal chance (ie. could have 20
        # of one step and 0 of another)
        if shuffle == 'random':
            for i in range(NUM_TOTAL_MOVES):
                game = random.choice(list(data_bank[agent]))
                step_num = random.randint(0, len(game[0])-1)
                agent_obs.append(game[0][step_num])
                agent_act.append(game[1][step_num])
        # append each observations/actions in order
        elif shuffle == 'false':
            for i in range(NUM_ADHOC_GAMES):
                game_len = len(data_bank[agent][i][0])
                for l in range(game_len):
                    agent_obs.append(list(data_bank[agent][i][0][l]))
                    agent_act.append(list(data_bank[agent][i][1][l]))
        # append each observation/action exactly once, but in random order
        elif shuffle == 'even':
            agent_obs_act = np.array()
            for i in range(NUM_ADHOC_GAMES):
                game_len = len(data_bank[agent][i][0])
                for l in range(game_len):
                    agent_obs_act.append(data_bank[agent][i][0][l] + data_bank[agent][i][1][l])
            for i in range(0, len(agent_obs_act), 2):
                agent_obs.append(agent_obs_act[i])
                agent_act.append(agent_obs_act[i+1])
        
        agent_obs = np.array(agent_obs)
        agent_act = np.array(agent_act)
        
        return agent_obs, agent_act

def main(args):
    data = Dataset()
    args = parse_args.resolve_datapath(args,
        data.game_type,
        data.num_players,
        data.num_unique_agents,
        data.num_games)

    try:
        raw_data = pickle.load(open(args.datapath, "rb"), encoding='latin1')

    except IOError:
        call("python create_data.py --datapath " + args.datapath, shell=True)
        raw_data = pickle.load(open(args.datapath, "rb"), encoding='latin1')
    
    data.read(raw_data)
    
   # import pdb; pdb.set_trace()
    return data


if __name__ == "__main__":
    args = parse_args.parse()
    args = parse_args.resolve_configpath(args)
    main(args)
