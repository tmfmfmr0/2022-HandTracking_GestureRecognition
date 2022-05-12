#%%
import numpy as np
import pandas as pd
import os

path = "./Dataset"
file_list = os.listdir(path)

#%%
# 
dfa = pd.read_csv(f'./Dataset/{file_list[0]}', header=None)
dfb = pd.read_csv(f'./Dataset/{file_list[1]}', header=None)
dfc = pd.read_csv(f'./Dataset/{file_list[2]}', header=None)

# %%
dfxa = dfa.iloc[:, :-1]
dfxb = dfb.iloc[:, :-1]
dfxc = dfc.iloc[:, :-1]

dfya = dfa.iloc[:, -1]
dfyb = dfb.iloc[:, -1]
dfyc = dfc.iloc[:, -1]

#%%

print(dfxa)
print(dfya)

# %%
