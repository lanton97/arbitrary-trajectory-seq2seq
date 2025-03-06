from simulations.bullet_sim.data_generator import *
import pandas as pd

num_runs = 22
seed = 203

df = pd.DataFrame()
offset = 0
# add on to the previous dataframe by downloading it first
try:
    df = pd.read_csv("datasets/bullet.csv", index_col=False)
    offset = df.tail(n=1)['Run']
    offset = offset.iloc[0] + 1
except:
    print('Fail')
    pass

try:
    for i in range(num_runs):
        print(seed)
        d = generateBulletRun(view=False, seed = seed) 
        d['Run'] = i + offset
        d.set_index(['Run', 'Step'])
        df = pd.concat([df,d])
        df.to_csv("datasets/bullet.csv", encoding='utf-8', index=False)
        seed += 1
except:
    print('Exit with Seed:' + str(seed))

print(seed)
print(df)
