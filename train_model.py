import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1
import torch
import os
import time
from rlcard.utils import (
  tournament,
  reorganize,
  Logger,
)

total_start_time = time.time()

ntrain = 250001 # total training games to play
ntest = 1000 # how many testing games to play for performance evaluation
testinterval = 50000 # how many training games to play between tests
updateinterval = 1000 # how many training games to play between opponent updates
checkpointinterval = 100 * updateinterval # checkpoints at a multiple of this aren't deleted
transition = 0 # how many training games to play before switching to self-play

# create environment
current_dir = os.getcwd()
env = rlcard.make('limit-holdem')
training_agent = NFSPAgent(num_actions=env.num_actions, state_shape=env.state_shape[0], hidden_layers_sizes=[128, 64, 128, 64], reservoir_buffer_capacity=600000, sl_learning_rate=0.01, q_mlp_layers=[128, 64, 128, 64], q_replay_memory_size=3000000, device=torch.device('cpu'), save_path=current_dir, save_every=testinterval)
opposing_agent = NFSPAgent(num_actions=env.num_actions, state_shape=env.state_shape[0], hidden_layers_sizes=[128, 64, 128, 64], reservoir_buffer_capacity=600000, sl_learning_rate=0.01, q_mlp_layers=[128, 64, 128, 64], q_replay_memory_size=3000000, device=torch.device('cpu'), save_path=current_dir, save_every=testinterval)

# original layer sizes
# training_agent = NFSPAgent(num_actions=env.num_actions, state_shape=env.state_shape[0], hidden_layers_sizes=[64, 64], reservoir_buffer_capacity=600000, sl_learning_rate=0.01, q_mlp_layers=[64, 64], q_replay_memory_size=30000000, device=torch.device('cpu'), save_path=current_dir, save_every=testinterval)

# for training vs rule bot and transitioning to self-play
# opposing_agent = LimitholdemRuleAgentV1()

# for resuming training from checkpoint
# checkpoint = torch.load('1000kv4.pt')
# training_agent = NFSPAgent.from_checkpoint(checkpoint)
# opposing_agent = NFSPAgent.from_checkpoint(checkpoint)

env.set_agents([training_agent, opposing_agent])

# run experiments and log progress
training_start_time = time.time()
with Logger("results/") as logger:
    for episode in range(ntrain):

        # select policy (best_response, average_policy)
        training_agent.sample_episode_policy()
        opposing_agent.sample_episode_policy()
        
        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)

        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        # Feed transitions into agent memories, and train both agents
        for ts in trajectories[0]:
          training_agent.feed(ts)
          opposing_agent.feed(ts)

        # evaluate performance against random agent
        if episode % testinterval == 0:
          # control_agent = LimitholdemRuleAgentV1()
          control_agent = RandomAgent(num_actions=env.num_actions)
          env.set_agents([training_agent, control_agent])
          logger.log_performance(episode, tournament(env, ntest)[0])
          env.set_agents([training_agent, opposing_agent])

        if episode % updateinterval == 0:
          print(f'\ntrained {updateinterval} episodes in {time.time() - training_start_time:.3f} seconds')
          training_start_time = time.time()
        
        # DISABLED
        # set opposing agent equal to training agent
        if episode >= transition and episode % updateinterval == 0 and 0:
          print(f'\ntrained {updateinterval} episodes in {time.time() - training_start_time:.3f} seconds. episode: {episode}')
          update_start_time = time.time()
          fname = 'temp/v4checkpoint' + str(episode) + '.pt'
          training_agent.save_checkpoint(path=current_dir, filename=fname)
          checkpoint = torch.load(fname)
          opposing_agent = NFSPAgent.from_checkpoint(checkpoint)
          env.set_agents([training_agent, opposing_agent])
          if episode % checkpointinterval != 0:
             os.remove(fname)
          print(f'opponent updated in {time.time() - update_start_time:.3f} seconds')
          training_start_time = time.time()

        # occasionally create a checkpoint
        if episode % checkpointinterval == 0:
          # check on training_agent vs opposing_agent performance
          train_vs_opp = tournament(env, ntest)[0]
          print(f'\ntraining vs opposing: {train_vs_opp:.3f}')
          update_start_time = time.time()
          tfname = 'temp/v5tcp' + str(episode) + '.pt'
          ofname = 'temp/v5ocp' + str(episode) + '.pt'
          training_agent.save_checkpoint(path=current_dir, filename=tfname)
          opposing_agent.save_checkpoint(path=current_dir, filename=ofname)
          print(f'checkpoints created in {time.time() - update_start_time:.3f} seconds')


    # Get the paths
    csv_path, fig_path = logger.csv_path, logger.fig_path 

    print('\ncsv_path:', csv_path, '\nfig_path:', fig_path)
    training_agent.save_checkpoint(path=current_dir, filename='trained_t_rename_me.pt')
    opposing_agent.save_checkpoint(path=current_dir, filename='trained_o_rename_me.pt')

print(f'finished in {time.time() - total_start_time:.3f} seconds')