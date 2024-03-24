from rlcard.agents import NFSPAgent
from collections import OrderedDict
import numpy as np
import torch

trained_model = NFSPAgent.from_checkpoint(checkpoint=torch.load('saved_model.pt'))

# get incoming state
sample_state1 = {
  'legal_actions': OrderedDict([(1, None), (2, None), (3, None)]),
  'obs': np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]),
  'raw_obs': {
    'hand': ['H4', 'S7'],
    'public_cards': ['D7', 'D2', 'SA', 'CJ', 'C4'],
    'all_chips': [16, 20], 'my_chips': 16,
    'legal_actions': ['raise', 'fold', 'check'],
    'raise_nums': [4, 1, 1, 1]},
  'raw_legal_actions': ['raise', 'fold', 'check'],
  'action_record': [(1, 'raise'), (0, 'raise'), (1, 'raise'), (0, 'raise'), (1, 'call'), (0, 'check'), (1, 'raise'), (0, 'call'), (1, 'raise'), (0, 'call'), (1, 'raise'), (0, 'fold')]
}

sample_state2 = {
  'legal_actions': OrderedDict([(1, None), (2, None), (3, None)]),
  'raw_legal_actions': ['raise', 'fold', 'check'],
  'obs': np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.])
}

# send response
action, info = trained_model.eval_step(sample_state1)
print(f'action: {action}\ninfo: {info}')
action, info = trained_model.eval_step(sample_state2)
print(f'action: {action}\ninfo: {info}')