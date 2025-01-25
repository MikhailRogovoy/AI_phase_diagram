# Импортируем необходимые модули
from pandas import read_csv
from numpy import loadtxt
from sklearn.model_selection import TunedThresholdClassifierCV, train_test_split
from xgboost import XGBClassifier

# !!!ОБУЧАЕМ МОДЕЛЬ!!!
# Загружаем файл с обработанным столбцом 'h' в 0 и 1 в NDArray
dataset_train_test = loadtxt('./data/Main_Data_Model_0_1.csv', delimiter=",", skiprows = 1)
# Разбиваем данные на признаки X и таргеты y
X = dataset_train_test[:,2:9]
y = dataset_train_test[:,1]
# Разбиваем данные на train и test выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = None)
# Подгоняем модель по данным X, y-train
model = XGBClassifier()
model.fit(X_train, y_train)

# !!!ПРЕДИКТ ПО НЕНАСТРОЕННОЙ МОДЕЛИ ПО СЭМПЛАМ!!!
# Импортируем датасет для предикта по ненастроенной по threshold модели из файла и указываем колонки признаков
file_name = 'Samples_For_Model_0_1_J1.csv'
dataset_predict = loadtxt('./Threshold_Selection_For_Samples_Model_0_1/' + file_name, delimiter=";", skiprows = 1)
X_predict = dataset_predict[:,2:9]
# Предикт таргета по ненастроенной по threshold модели
predict_target = model.predict_proba(X_predict)
predict_target = [round(value) for value in predict_target[:, 1]]
# Записываем предикт из ненастроенной по threshold модели
dataframe = read_csv('./Threshold_Selection_For_Samples_Model_0_1/' + file_name)
dataframe['predict'] = predict_target
dataframe.to_csv('./Threshold_Selection_For_Samples_Model_0_1/' + 'Predict_' + file_name, index = False, sep=',', float_format='%.6f')

# !!!ПРЕДИКТ ПО НАСТРОЕННОЙ МОДЕЛИ ПО СЭМПЛАМ!!!
# Создаем и настраиваем модель по обучающей выборке с лучшим threshold
model_tuned = TunedThresholdClassifierCV(model, scoring = "accuracy")
model_tuned.fit(X_train, y_train)
# Показываем лучшее значение threshold
print(f"The best threshold is {model_tuned.best_threshold_:.3f}")
# Предикт таргета по настроенной по threshold модели
y_pred_tuned = model_tuned.predict_proba(X_predict)
predictions_tuned = [round(value) for value in y_pred_tuned[:, 1]]
# Записываем предикт из настроенной по threshold модели
dataframe_tuned = read_csv('./Threshold_Selection_For_Samples_Model_0_1/' + 'Predict_' + file_name)
dataframe_tuned['predict_tuned'] = predictions_tuned
dataframe_tuned.to_csv('./Threshold_Selection_For_Samples_Model_0_1/' + 'Predict_' + file_name, index = False, sep=',', float_format='%.6f')
# СМОТРИМ РЕЗУЛЬТАТ В ФАЙЛЕ

# !!!ПРЕДИКТ ВРУЧНУЮ ПО СЭМПЛАМ!!!
# Предикт таргета по threshold вручную
threshold = 0.59
y_pred_by_hand = model.predict_proba(X_predict)
predictions_by_hand = []
for value in y_pred_tuned[:, 1]:
    if value > threshold:
        predictions_by_hand.append(round(value))
    else:
        predictions_by_hand.append(0)
# Записываем предикт из настроенной по threshold модели
dataframe_by_hand = read_csv('./Threshold_Selection_For_Samples_Model_0_1/' + 'Predict_' + file_name)
dataframe_by_hand['predict_by_hand'] = predictions_by_hand
dataframe_by_hand.to_csv('./Threshold_Selection_For_Samples_Model_0_1/' + 'Predict_' + file_name, index = False, sep=',', float_format='%.6f')
# СМОТРИМ РЕЗУЛЬТАТ В ФАЙЛЕ