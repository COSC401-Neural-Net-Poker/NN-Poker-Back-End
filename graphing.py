import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/performance.csv')
plt.plot(df['episode'], df['reward'])
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.savefig('results/rewardbyepisode.png')