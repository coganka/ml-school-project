import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

def custom_data_test():
    new_test_data = pd.read_csv('data/custom.csv')
    new_test_data['Target'] = new_test_data['Close'].shift(-1)  # Predicting the next day's closing price

    new_test_data = new_test_data[:-1]
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    svr_model = load('SVR_model.joblib')
    gb_model = load('Gradient Boosting Regressor_model.joblib')
    rf_model = load('Random Forest Regressor_model.joblib')

    # Extract features from the new test data
    X_new_test = new_test_data[features]  # Ensure these features are processed the same way as training data
    y_new_test = new_test_data['Target']

    # Use the loaded model to make predictions
    svr_predictions = svr_model.predict(X_new_test)
    gb_predictions = gb_model.predict(X_new_test)
    rf_predictions = rf_model.predict(X_new_test)


    plt.figure(figsize=(8, 6))

    # Plot SVR predictions
    plt.subplot(3, 1, 1)
    plt.plot(y_new_test.reset_index(drop=True), label='Actual', color='blue')
    plt.plot(svr_predictions, label='SVR Predicted', color='red', alpha=0.7)
    plt.title('SVR Model Predictions')
    plt.legend()

    # Plot Gradient Boosting predictions
    plt.subplot(3, 1, 2)
    plt.plot(y_new_test.reset_index(drop=True), label='Actual', color='blue')
    plt.plot(gb_predictions, label='Gradient Boosting Predicted', color='red', alpha=0.7)
    plt.title('Gradient Boosting Model Predictions')
    plt.legend()

    # Plot Random Forest predictions
    plt.subplot(3, 1, 3)
    plt.plot(y_new_test.reset_index(drop=True), label='Actual', color='blue')
    plt.plot(rf_predictions, label='Random Forest Predicted', color='red', alpha=0.7)
    plt.title('Random Forest Model Predictions')
    plt.legend()

    plt.tight_layout()
    plt.show()
