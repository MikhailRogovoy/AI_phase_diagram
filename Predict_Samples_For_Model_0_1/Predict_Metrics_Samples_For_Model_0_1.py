import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных с предсказаниями
new_data = pd.read_csv('./Predict_Samples_For_Model_0_1/Predict_Samples_For_Model_0_1_J-1.csv')

# Получаем уникальные значения семплов
samples = new_data['sample'].unique()

# Словарь для хранения метрик для каждого семпла
metrics = {
    'Sample': [],
    'Confusion Matrix': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Predicted_Target': []  # Добавим Predicted_Target в метрики
}

# Рассчитываем метрики для каждого семпла
for sample in samples:
    # Фильтруем данные по текущему семплу
    sample_data = new_data[new_data['sample'] == sample]

    # Истинные метки и предсказания
    y_true = sample_data['Predicted_Target']  # Истинные метки (target)
    y_pred = sample_data['Predicted_Target']  # Предсказания (если они есть)

    # Вычисляем полную матрицу ошибок (с учетом всех классов)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Указываем все возможные классы (0 и 1)

    # Метрики для текущего семпла
    accuracy = accuracy_score(y_true, y_pred)  # Точность
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1,
                                zero_division=1)  # Precision для класса 1
    recall = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=1)  # Recall для класса 1
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=1)  # F1 для класса 1

    # Добавляем в словарь
    metrics['Sample'].append(sample)
    metrics['Confusion Matrix'].append(cm.tolist())  # Преобразуем в список для удобства
    metrics['Accuracy'].append(accuracy)
    metrics['Precision'].append(precision)
    metrics['Recall'].append(recall)
    metrics['F1 Score'].append(f1)
    metrics['Predicted_Target'].append(y_pred.iloc[0])  # Добавим Predicted_Target для каждого семпла

# Создаем DataFrame для вывода метрик
metrics_df = pd.DataFrame(metrics)

# Выводим метрики
print(metrics_df)

# Сохраняем метрики в CSV файл
metrics_df.to_csv('./Predict_Samples_For_Model_0_1/Predict_Metrics_Samples_For_Model_0_1_J-1.csv', index=False)

# Сохраняем метрики в текстовый файл с матрицей в виде 2 строк
with open('./Predict_Samples_For_Model_0_1/Predict_Metrics_Samples_For_Model_0_1_J-1.txt', 'w') as txt_file:
    txt_file.write("Sample Metrics Report\n")
    txt_file.write("=" * 50 + "\n")

    for index, row in metrics_df.iterrows():
        txt_file.write(f"Sample {row['Sample']}\n")

        # Выводим confusion matrix в виде 2 строк
        cm = row['Confusion Matrix']
        txt_file.write("Confusion Matrix:\n")
        txt_file.write(f"{cm[0][0]}  {cm[0][1]}\n")  # Первая строка
        txt_file.write(f"{cm[1][0]}  {cm[1][1]}\n")  # Вторая строка

        txt_file.write(f"Predicted Target: {row['Predicted_Target']}\n")
        txt_file.write("-" * 50 + "\n")

    txt_file.write("=" * 50 + "\n")
    txt_file.write("End of Report\n")