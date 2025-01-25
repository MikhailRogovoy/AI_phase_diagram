# AI_phase_diagram
Результаты работ по теме "Интеллектуальное материаловедение. Определение численной версии по фазовой диаграмме"

# Описание структуры проекта
Весь код проекта представлен в файлах .py и дублирующие их .ipynb

Папка data
----------------
Содержит исходные файлы данных:
- Main_Data_Model_0_1.csv - Тренировочные и тестовые данные для классификации модели 0-1.  
  - Столбец h разделяет классы:  
    - 0 — нулевая версия численной модели Изинга.  
    - 1 — первая версия численной модели Изинга.

- Samples_For_Model_0_1_J1.csv - Слепые данные (J1) для классификации модели 0-1.  
  - Нет целевого признака h.  
  - Столбец sample группирует строки одного целевого признака.

- Samples_For_Model_0_1_J-1.csv - Слепые данные (J-1) для классификации модели 0-1.  
  - Аналогично файлу J1.

- Main_Data_Model_0_2_320_Samples.csv - Тренировочные и тестовые данные для классификации модели 0-2.  
  - Столбец Class присваивает каждой строке класс 0 или 1.  
  - Разделение производится по значению 0,0125.

- Keys_For_Model_0_2.csv - Файл с ключами для каждого семпла.

Папка Metrics_For_Model_0_1
-------------------------------
Содержит коды для вычисления метрик классификации модели 0-1:
- ML_Model_0_1_Selection.ipynb / ML_Model_0_1_Selection.py - Вывод метрик для различных моделей машинного обучения.  
  - Вход: Main_Data_Model_0_1.csv  
  - Выход: Метрики.

- Full_Metrics_For_Model_0_1_XGBoost.ipynb / Full_Metrics_For_Model_0_1_XGBoost.py - Метрики для модели XGBoost (включая кросс-валидацию, SHAP и важность признаков).  
  - Вход: Main_Data_Model_0_1.csv  
  - Выход: Метрики, кросс-валидация, SHAP.

- DecisionTree_Visual_For_0_1.ipynb / DecisionTree_Visual_For_0_1.py - Визуализация дерева решений (параметры Gini и Entropy).  
  - Вход: Main_Data_Model_0_1.csv  
  - Выход: Полные и упрощенные деревья решений.

Папка Predict_Samples_For_Model_0_1
---------------------------------------
Содержит коды для предсказания классификации модели 0-1:
- Predict_Samples_For_Model_0_1.ipynb / Predict_Samples_For_Model_0_1.py - Добавляет колонки Predicted_Target и Predict_Proba в слепые данные.  
  - Вход: Main_Data_Model_0_1.csv, Samples_For_Model_0_1_J1.csv, или Samples_For_Model_0_1_J-1.csv  
  - Выход: Predict_Samples_For_Model_0_1_J1.csv или Predict_Samples_For_Model_0_1_J-1.csv.

- Predict_Metrics_Samples_For_Model_0_1.ipynb / Predict_Metrics_Samples_For_Model_0_1.py - Вывод метрик по семплам с сохранением результатов в .txt и .csv.  
  - Вход: Main_Data_Model_0_1.csv, Samples_For_Model_0_1_J1.csv, или Samples_For_Model_0_1_J-1.csv  
  - Выход: Метрики для семплов.

Папка Threshold_Selection_For_Samples_Model_0_1
---------------------------------------------------
Содержит коды для выбора порога (threshold):
- Threshold_Selection_For_Samples_Model_0_1.ipynb / Threshold_Selection_For_Samples_Model_0_1.py - Добавляет столбцы с предсказаниями (predict, predict_tuned, predict_by_hand).  
  - Вход: Main_Data_Model_0_1.csv, Samples_For_Model_0_1_J1.csv, или Samples_For_Model_0_1_J-1.csv  
  - Выход: Predict_Samples_For_Model_0_1_J1.csv или Predict_Samples_For_Model_0_1_J-1.csv.

Папка Drop_Column_For_Model_0_1
-----------------------------------
Содержит коды для использования определенных признаков:
- Full_Metrics_Xfm_C_For_Model_0_1.ipynb / Full_Metrics_Xfm_C_For_Model_0_1.py - Оставляет только Xfm (магнитная восприимчивость) и C (теплоемкость).  
  - Вход: Main_Data_Model_0_1.csv  
  - Выход: Метрики, кросс-валидация, SHAP.

- Full_Metrics_Fm_C_For_Model_0_1.ipynb / Full_Metrics_Fm_C_For_Model_0_1.py - Оставляет только Fm (ферромагнитная намагниченность) и C.  
  - Вход: Main_Data_Model_0_1.csv  
  - Выход: Метрики, кросс-валидация, SHAP.

Папка Metrics_For_Model_0_2
-------------------------------
Аналогична папке Metrics_For_Model_0_1, но для модели 0-2.

Папка Drop_Column_For_Model_0_2
-----------------------------------
Аналогична папке Drop_Column_For_Model_0_1, но для модели 0-2.

Папка Hyperparameters_For_Model_0_2
---------------------------------------
Содержит код для тестирования гиперпараметров модели 0-2.

Папка Fast_Check_Metrics
----------------------------
Коды для вывода упрощенных метрик для моделей 0-1 и 0-2:
- Вход: Standart_Model_0_1.csv или Standart_Model_0_2.csv.  
- Выход: Метрики.

Папка Find_Best_Splitting_Data
----------------------------------
Содержит код для поиска наилучшего разделения классов:
- Find_Best_Splitting_Data_For_Models.ipynb / Find_Best_Splitting_Data_For_Models.py - Определяет значение поля для разделения классов.  
  - Вход: initial_data_0_1.csv или initial_data_0_2.csv.  
  - Выход: График ROC-AUC.
