import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents import RandomAgent
import torch
import os
import time
from rlcard.utils import (
  tournament,
  reorganize,
  Logger,
)

start_time = time.time()

ntrain = 100001 # total training games to play
ntest = 1000 # how many testing games to play for performance evaluation
testinterval = 1000 # how many training games to play before testing
updateinterval = 2000 # how many training games to play before updating opponent

# create environment
current_dir = os.getcwd()
env = rlcard.make('limit-holdem')
training_agent = NFSPAgent(state_shape=[72], hidden_layers_sizes=[64,64], q_mlp_layers=[64,64], device=torch.device('cpu'), save_path=current_dir, save_every=1)
opposing_agent = NFSPAgent(state_shape=[72], hidden_layers_sizes=[64,64], q_mlp_layers=[64,64], device=torch.device('cpu'), save_path=current_dir, save_every=1)
env.set_agents([training_agent, opposing_agent])

# run experiments and log progress
with Logger("results/") as logger:
    for episode in range(ntrain):

        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)

        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
          training_agent.feed(ts)

        # evaluate performance against random agent
        if episode % testinterval == 0:
          control_agent = RandomAgent(num_actions=env.num_actions)
          env.set_agents([training_agent, control_agent])
          logger.log_performance(episode, tournament(env, ntest)[0])
          env.set_agents([training_agent, opposing_agent])
        
        # set opposing agent equal to training agent
        if episode % updateinterval == 0:
          fname = 'temp/tempcheckpoint' + str(episode) + '.pt'
          training_agent.save_checkpoint(path=current_dir, filename=fname)
          checkpoint = torch.load(fname)
          opposing_agent = NFSPAgent.from_checkpoint(checkpoint)
          env.set_agents([training_agent, opposing_agent])

    # Get the paths
    csv_path, fig_path = logger.csv_path, logger.fig_path

    print('\ncsv_path:', csv_path, '\nfig_path:', fig_path)
    training_agent.save_checkpoint(path=current_dir, filename='trained_checkpoint0.pt')

print(f'finished in {time.time() - start_time:.3f} seconds')