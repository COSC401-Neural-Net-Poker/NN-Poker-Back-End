import rlcard
from rlcard.agents import NFSPAgent
import torch
from rlcard.utils import (
  tournament,
  Logger
)

testpath = 'model10k.pt'
controlpath = 'saved_model.pt'
ngames = 10000

# create environment
env = rlcard.make('limit-holdem')
test_agent = NFSPAgent.from_checkpoint(torch.load(testpath))
control_agent = NFSPAgent.from_checkpoint(torch.load(controlpath))
env.set_agents([test_agent, control_agent])

result = tournament(env, ngames)[0]
print(f'Average payoff for {testpath} against {controlpath} over {ngames} games: {result}')