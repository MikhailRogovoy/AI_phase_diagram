import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def get_roc_auc(model, X_test, y_test, i):

    y_pred = model.predict_proba(X_test)
    y_pred = [round(value) for value in y_pred[:, 1]]
    roc_auc = roc_auc_score(y_test, y_pred)
    threshold_h = i/100000
    return threshold_h, roc_auc


def get_best_threshold_h(model_num=1):

    roc_auc_list = list()
    errors_list = list()
    for i in range (400000, 497600, 2000):
        filename = "./Find_Best_Splitting_Data/initial_data_0_1.csv"
    # Выбор данных для модели
        if(model_num == 2):
            filename = "./Find_Best_Splitting_Data/initial_data_0_2.csv"
    # Загружаем файл
        data = pd.read_csv(filename, sep=',', encoding='utf-8')
        X = data[["E", "C", "FM", "Xfm", "AFM", "Xafm"]]
        data['h'] = data['h'].apply(lambda h_value: 0 if h_value < (i/100000) else 1)
        y = data["h"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Обучаем модель по данным X_train, y-train
        model = XGBClassifier()
        model.fit(X_train, y_train)
        precision, error = get_roc_auc(model, X_test, y_test, i)
        roc_auc_list.append(precision)
        errors_list.append(error)

    plt.style.use('ggplot')
    plt.plot(roc_auc_list, errors_list, marker='o', markersize=8, linestyle='-',
             linewidth=1, label=f'Model 0-{model_num}')
    plt.title('ROC AUC values vs h threshold')
    plt.xlabel('Threshold h')
    plt.ylabel('ROC AUC value')
    plt.legend()
    plt.show()


get_best_threshold_h(1)