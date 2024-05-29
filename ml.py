# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import dataset
df = pd.read_csv('./data-2.csv')

categorical_columns = ['game_phase']
df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')

df_encode.columns = df_encode.columns.str.strip()

def train_models():
    X = df_encode['material_balance']
    y = df_encode['evaluation']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)
    
    X_train = np.array(X_train).reshape(-1,1)
    X_test = np.array(X_test).reshape(-1,1)

    reg=LinearRegression(fit_intercept=True)
    reg.fit(X_train,y_train)
    c = reg.intercept_
    m = reg.coef_
    
    return m[0], c    # Y = mX + c
    
def evaluate_board(material):
    m, c = train_models()
    return m*material + c