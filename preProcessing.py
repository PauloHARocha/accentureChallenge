import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns;sns.set(style='whitegrid')
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# The objective of this module is to prepare the data for further analyses
# On the explanation.ipynb file you can better visualize the transformations in the data  

info = pd.read_csv('heroes_information.csv', index_col=[0]) # index_col=[0] to get rid of first column

# Duplications were visualized in the heroes_information data, so it was chosen to remove these duplications
info = info.drop_duplicates(subset='name', keep='first')

power = pd.read_csv('super_hero_powers.csv')

# Because the datasets have a different number of heroes, the intersection between the two based on the heroes name was made 
names_info = set(info['name'])
names_power = set(power['hero_names'])

intersec = list(names_info.intersection(names_power))

df1 = info[info['name'].isin(intersec)]
df2 = power[power['hero_names'].isin(intersec)]

df2.rename(columns={'hero_names':'name'}, inplace=True)

# Merge both datasets by name
info_power = pd.merge(df1, df2, on="name")

# Save merged dataset
info_power.to_csv('info_power.csv', index=False)




