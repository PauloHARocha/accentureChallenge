import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# The objective of this module is to prepare the data for further analyses
# On the explanation.ipynb file you can better visualize the transformations in the data  

info = pd.read_csv('datasets/heroes_information.csv', index_col=[0]) # index_col=[0] to get rid of first column

# Duplications were visualized in the heroes_information data, so it was chosen to remove these duplications
info = info.drop_duplicates(subset='name', keep='first')

power = pd.read_csv('datasets/super_hero_powers.csv')

# Because the datasets have a different number of heroes, the intersection between the two based on the heroes name was made 
names_info = set(info['name'])
names_power = set(power['hero_names'])

intersec = list(names_info.intersection(names_power))

df1 = info[info['name'].isin(intersec)]
df2 = power[power['hero_names'].isin(intersec)]

df2.rename(columns={'hero_names':'name'}, inplace=True)

# Merge both datasets by name
info_power = pd.merge(df1, df2, on="name")

# After merging the two datasets, the next step is to clear the missing values
# You can visualize the features in the explanation file

# Replacing invalid values '-' and -99 for NaN
info_power = info_power.replace('-', np.nan).replace(-99, np.nan)

# Encoder for categorical and boolean data
le = preprocessing.LabelEncoder()

# Gender - Just removing missing values
feature = 'Gender'
info_power = info_power.dropna(subset=[feature])
info_power[feature] = le.fit_transform(info_power[feature])

# In the next features, the categories that contained more information (more heroes) 
# were kept and the rest along with the missing values ​​were grouped into a category 'other (name of feature)'.
# Then each new category were transformed in a column of True or False, to avoid bias with euclidian distance
# Eye color / Race / Hair color / Publisher 
features = ['Eye color', 'Race', 'Hair color', 'Publisher']
new_columns = [['blue', 'brown', 'green', 'red'],
               ['Human', 'Mutant'], 
               ['Black', 'Blond', 'Brown', 'No Hair', 'Red'], 
               ['Marvel Comics', 'DC Comics']]

for feature, keep_features in zip(features, new_columns):
    other_feature = f'other {feature}'
  
    # Create the new categories
    info_power[feature] = info_power[feature].apply(
        lambda x: other_feature if x not in keep_features else x)
    
    # Create a new column for each category
    feat_column = info_power.pop(feature) 
    keep_features += [other_feature]

    for kf in keep_features:
        info_power[kf] = (feat_column == kf)*1.0

# In the next features the undefined values ​​were redefined with median of the feature data
# Then normalized between 0 and 1, to avoid bias with euclidian distance
# Height / Weight
features = ['Height', 'Weight']
for feature in features:
    # Redefine missing values with the median
    info_power[feature] = info_power[feature].replace(np.nan, info_power[feature].median())
    
    # Normalize 
    std = MinMaxScaler()
    info_power[feature] = std.fit_transform(info_power[[feature]])
                                                          
# Skin color - Because of the large number of missing data this column was dropped
feature = 'Skin color'
info_power = info_power.drop(columns=feature)

# Alignment - For this feature the missing values and the neutral category were removed
feature = 'Alignment'

info_power = info_power.dropna(subset=[feature])
info_power = info_power[info_power[feature] != 'neutral']

info_power[feature] = le.fit_transform(info_power[feature])


# For the heroes super powers the boolean data were changed to float
# Heroes super powers
power_cols = power.columns[1:].values # Get power columns except the name column

info_power[power_cols] = info_power[power_cols].astype(float)

# Save pre-processed dataset
info_power.to_csv('datasets/info_power_processed.csv', index=False)
print('The pre-processed dataset was created as info_power_processed.csv')
print(f'This dataset has {info_power.shape[0]} rows and {info_power.shape[1]} columns')