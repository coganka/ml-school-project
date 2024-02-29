import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from custom_test import custom_data_test
from scrapper import get_data
import numpy as np

class StockPredictionModel:
    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = {
            'SVR': SVR(kernel="rbf", gamma="scale"),
            'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=random_state),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=random_state)
        }
        self.results = {}
        self.last_fold_predictions = {} #storing dict (pred and real data) for plotting

    #train and pedict using k-fold and store after each fold but update it dont create new
    def train_and_evaluate(self):
        self.results = {name: [] for name in self.models.keys()}
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_index, test_index in kf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            for name, model in self.models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                if name not in self.results:
                    self.results[name] = []
                self.results[name].append(rmse)
                self.last_fold_predictions[name] = (y_test, y_pred) #update for the latest fold

        # average RMSE 
        self.avg_results = {name: np.mean(scores) for name, scores in self.results.items()}
        return self.avg_results
    

    # plotting with matplotlib
    def plot_predictions(self):
        plt.figure(figsize=(8, 2 * len(self.models)))

        for i, (name, (y_actual, y_pred)) in enumerate(self.last_fold_predictions.items(), 1):
            plt.subplot(len(self.models), 1, i)
            plt.plot(y_actual.reset_index(drop=True), label='Actual', color='blue')
            plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
            plt.title(f'{name} Predictions')
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.legend()

        plt.tight_layout()
        plt.show()

    
    def get_results(self):
        return self.results



#get input and set stock data location
while True:
    user_input = input('Run web Scrapper and get scrapped data ? [y/n]')
    if user_input.lower() == 'y':
        get_data()
        file_path = 'data.csv'
        break
    elif user_input.lower() == 'n':
        file_path = 'data/MSFT.csv'
        break
    else:
        continue

stock_data = pd.read_csv(file_path)

#preprocessing the data 
stock_data['Target'] = stock_data['Close'].shift(-1)  
stock_data = stock_data[:-1]

# 6 features
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
X = stock_data[features]
y = stock_data['Target']

#initialize
trained_model = StockPredictionModel(X, y)

#train and evaluate the models
trained_model.train_and_evaluate()

#Get results
results = trained_model.get_results()
print(results)

#log RMSE results
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'RMSE'])
print(results_df)

#save models
for name, model in trained_model.models.items():
    dump(model, f'{name}_model.joblib')

#plot
trained_model.plot_predictions()


custom_data_test()