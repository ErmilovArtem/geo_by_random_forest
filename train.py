import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('train_preprocessing.csv')
df = df.drop(columns=['Unnamed: 0'])

with open('kmeans.pkl', 'rb') as f:
    loaded_kmeans_model = pickle.load(f)

# Разделение данных на признаки и целевую переменную
X = df.drop(columns=['score'])
y = df['score']

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()

model.fit(X_train, y_train)
joblib.dump(model, 'RandomForestRegressor.pkl')

y_pred = model.predict(X_test)

if __name__ == '__main__':
    print(str(model)[:30] + " Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print(str(model)[:30] + " Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print(str(model)[:30] + " MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print(str(model)[:30] + " MPAPE:", mean_absolute_percentage_error(y_test, y_pred) / np.mean(y_test))

    # Визуализация фактических и предсказанных значений
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[100:200], label='Фактические значения', marker='o')
    plt.plot(y_pred[100:200], label='Предсказанные значения', marker='x')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.title('Фактические vs Предсказанные значения - ' + str(model)[:30])
    plt.legend()
    plt.show()