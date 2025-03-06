from simulations.data_gen import *
import pandas as pd

num_runs = 100

generator = convoyDatasetGenerator()

df = pd.DataFrame()
offset = 0
# add on to the previous dataframe by downloading it first
try:
    df = pd.read_csv("datasets/convoyDS.csv", index_col=False)
    offset = df.tail(n=1)['Run']
    offset = offset.iloc[0] + 1
except:
    print('Fail')
    pass

for i in range(num_runs):
    generator.reset()
    d, t = generator.generateRun()
    d['Run'] = i + offset
    d.set_index(['Run', 'Step'])
    df = pd.concat([df,d])
    df.to_csv("datasets/convoyDS.csv", encoding='utf-8', index=False)

print(df)
