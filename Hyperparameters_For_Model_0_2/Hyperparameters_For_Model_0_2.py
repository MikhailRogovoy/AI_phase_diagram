import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score, precision_recall_curve, auc)
import numpy as np

# Подгружаем наш датасет
data = pd.read_csv('./data/Main_Data_Model_0_2_320_Samples.csv', delimiter=';')

# Разделение данных
Y = data['Class']  # Целевая переменная
X = data[['Specific_Heat', 'Magnetization', 'FM_susc', 'AFM_vector', 'AFM_susc']]  # Признаки

# Инициализация k-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Метрики
accuracies, roc_aucs, pr_aucs = [], [], []
precision_zeros, precision_ones = [], []
recall_zeros, recall_ones = [], []
f1_zeros, f1_ones = [], []
confusion_matrices = []

# Для расчета метрики для дисбалансированных данных
threshold = 0.5  # Порог классификации, можно настраивать

# Инициализация модели XGBoost с возможными параметрами
scale_pos_weight = len(Y) / sum(Y == 0)  # Рассчитываем вес для редкого класса
model = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,  # Помощь при дисбалансе классов
    max_depth=3,  # Настроить по необходимости
    learning_rate=0.05,  # Настроить по необходимости
    n_estimators=100,  # Настроить по необходимости
    subsample=0.8,  # Настроить по необходимости
    colsample_bytree=0.8,  # Настроить по необходимости
    gamma=1,  # Регуляризация, настроить по необходимости
)

# Модель и оценка
for train_index, test_index in kfold.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Обучение модели
    model.fit(X_train, y_train)

    # Получаем вероятности для каждого класса
    y_pred_probs = model.predict_proba(X_test)[:, 1]

    # Настройка порога классификации (можно изменять порог)
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Оценка модели
    accuracies.append(accuracy_score(y_test, y_pred))
    roc_aucs.append(roc_auc_score(y_test, y_pred_probs))

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    pr_aucs.append(auc(recall, precision))

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

    # Для класса 0
    precision_zeros.append(precision_score(y_test, y_pred, pos_label=0))
    recall_zeros.append(recall_score(y_test, y_pred, pos_label=0))
    f1_zeros.append(f1_score(y_test, y_pred, pos_label=0))

    # Для класса 1
    precision_ones.append(precision_score(y_test, y_pred, pos_label=1))
    recall_ones.append(recall_score(y_test, y_pred, pos_label=1))
    f1_ones.append(f1_score(y_test, y_pred, pos_label=1))

# Результаты
print(f'Average Accuracy: {np.mean(accuracies):.2f}')
print(f'Average ROC-AUC: {np.mean(roc_aucs):.2f}')
print(f'Average Precision-Recall AUC: {np.mean(pr_aucs):.2f}')
print(f'Average Precision (Class 0): {np.mean(precision_zeros):.2f}')
print(f'Average Precision (Class 1): {np.mean(precision_ones):.2f}')
print(f'Average Recall (Class 0): {np.mean(recall_zeros):.2f}')
print(f'Average Recall (Class 1): {np.mean(recall_ones):.2f}')
print(f'Average F1-score (Class 0): {np.mean(f1_zeros):.2f}')
print(f'Average F1-score (Class 1): {np.mean(f1_ones):.2f}')

# Выводим confusion matrix для каждого фолда
for i, cm in enumerate(confusion_matrices):
    print(f'Fold {i+1} Confusion Matrix:\n{cm}')

thresholds_to_test = np.linspace(0, 1, 20)
for threshold in thresholds_to_test:
    y_pred = (y_pred_probs >= threshold).astype(int)
    print(f'Threshold: {threshold:.2f}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1: {f1_score(y_test, y_pred)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')