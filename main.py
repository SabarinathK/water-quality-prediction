from sqlite3 import DateFromTicks
import pandas
from sklearn.datasets import clear_data_home
from src.data_preprcessing import data_clean
from src.imbalance import  data_balance
from src.model import algo
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df=pandas.read_csv('water_potability.csv')
cleaned_data=data_clean(df)
print(cleaned_data)
data_balance()
algo(RandomForestClassifier())

