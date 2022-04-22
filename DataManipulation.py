import pandas as pd
#######################################################
# Initial Data Manipulation
#######################################################

# Load data file (South African Heart Disease dataset)
df = pd.read_csv('SAheart.csv')
df = df.drop('row.names', 1)  # erases column 'row.names'

# Set discrete variables to categorical type
df.loc[df['famhist'] == 'Present', 'famhist'] = int(1)
df.loc[df['famhist'] == 'Absent', 'famhist'] = int(0)
df = df.astype({"famhist": int})

# Define attribute names
attribute_names = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']
class_name = ['chd']

# Standardize Data
df2 = df.copy()
df2[attribute_names] = (df2[attribute_names] - df2[attribute_names].mean(0)) / df2[attribute_names].std(0)

# Data
X = df2[attribute_names].values
y = df2.values[:,-1]

