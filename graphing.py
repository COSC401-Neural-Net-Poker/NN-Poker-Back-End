import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv('results/performance.csv')
plt.plot(df['episode'], df['reward'])
plt.xlabel('step')
plt.ylabel('reward')
plt.grid()

if len(sys.argv) > 1:
  rl_rate = sys.argv[1]
  sl_rate = sys.argv[2]
  mem_init = sys.argv[3]
  eps_start = sys.argv[4]
  eps_steps = sys.argv[5]
  num = sys.argv[6]
  plt.title(f'rl_rate={rl_rate}, sl_rate={sl_rate}, mem_init={mem_init},\neps_start={eps_start}, eps_steps={eps_steps}')
  plt.savefig(f'results/rewardbyepisode{num}.png')
else:
  plt.savefig(f'results/rewardbyepisode.png')