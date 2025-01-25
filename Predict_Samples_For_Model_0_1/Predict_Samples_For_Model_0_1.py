import pandas as pd
from xgboost import XGBClassifier

# Загрузка старого файла с данными для обучения
data = pd.read_csv('./data/Main_Data_Model_0_1.csv')  # Данные с целевой переменной
Y = data.iloc[:, 1]  # Целевая переменная (второй столбец)
X = data.iloc[:, 2:9]  # Признаки (остальные столбцы)

# Загрузка нового файла без целевой переменной
new_data = pd.read_csv('./data/Samples_For_Model_0_1_J-1.csv', delimiter=';')  # Данные без целевой переменной

# Проверка структуры данных
print("Shape of training data:", X.shape)
print("Shape of new data:", new_data.shape)

# Выбор признаков с 3 по 9 колонку
new_data_features = new_data.iloc[:, 2:9]  # Признаки для предсказания

# Обучение модели на всех данных
model = XGBClassifier(random_state=42)
model.fit(X, Y)

# Предсказание для нового файла
new_data_probs = model.predict_proba(new_data_features)[:, 1]  # Вероятности для класса 1
new_data_predictions = (new_data_probs >= 0.5).astype(int)  # Преобразование в классы

# Добавление предсказаний и вероятностей в файл
new_data['Predicted_Target'] = new_data_predictions  # Столбец с предсказаниями
new_data['Predict_Proba'] = new_data_probs  # Столбец с вероятностями класса 1

# Сохранение результата в новый CSV файл
new_data.to_csv('./Predict_Samples_For_Model_0_1/Predict_Samples_For_Model_0_1_J-1.csv', index=False, float_format='%.6f')

print("Предсказания и вероятности добавлены и сохранены в 'Predicted_Samples_For_Model_0_1_J-1.csv'.")