import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')

head = df.head()
print(head)

df.info()
desc = df.describe()
print(desc)