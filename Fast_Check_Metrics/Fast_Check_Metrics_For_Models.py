import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import shap
from sklearn import tree


def get_roc_auc_curve(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)
    y_pred = [round(value) for value in y_pred[:, 1]]
    auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    # Строим график
    plt.style.use('ggplot')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, label=f"ROC curve (area={auc:.4f})" )
    plt.title('ROC-curve')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend()
    plt.show()
    print(f'AUC value{auc}')
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

def get_cross_val_score(model, X, y, cross_val_iterations):

    kfold = StratifiedKFold(n_splits=cross_val_iterations, random_state = None, shuffle = True)
    score_results = cross_validate(model, X, y, cv=kfold, scoring=["precision", "recall", "f1", "roc_auc"])
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    folds = range(1, cross_val_iterations + 1)
    for key in list(score_results.keys())[2:]:
        plt.plot(folds, score_results[key], marker='o', markersize=8, linestyle='-',
                 linewidth=1, label=key)
        for i, score in enumerate(score_results[key]):
            plt.text(folds[i], score, f'{score:.3f}', ha='center', fontsize=10, color='black')
    plt.xticks(folds)
    plt.xlabel('Fold Number', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.title('Cross-Validation scores', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()


def get_feature_importance(model):
    plt.figure(figsize=(10, 6))
    plt.barh(model.feature_names_in_, model.feature_importances_)
    plt.title('Feature Importance')
    plt.xlabel('Importance score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


def get_shap_plot(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)


def get_decision_tree_curve(X, y, *, tree_depth):
    clf_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=tree_depth)
    clf_gini.fit(X, y)
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf_gini, filled=True, feature_names=X.columns, class_names=['Class 0', 'Class 1'])
    plt.title(f'Decision Tree (depth = {tree_depth})')
    plt.show()


def get_model(model=1, *, roc_auc_curve=True, cross_val_score=True, cross_val_iterations=5,
              feature_importance=True, shap_plot=False, decision_tree_curve=False, tree_depth=3):
    filename = "./Fast_Check_Metrics/Standart_Model_0_1.csv"
    # Выбор данных для модели
    if(model == 2):
        filename = "./Fast_Check_Metrics/Standart_Model_0_2.csv"
    # Загружаем файл
    data = pd.read_csv(filename, sep=',', encoding='utf-8')
    X = data[["E", "C", "FM", "Xfm", "AFM", "Xafm"]]
    y = data["h"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Обучаем модель по данным X_train, y-train
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # Расчет метрик и построение графиков
    if(roc_auc_curve == True):
        get_roc_auc_curve(model, X_test, y_test)
    if(cross_val_score == True):
        get_cross_val_score(model, X, y, cross_val_iterations)
    if(feature_importance == True):
        get_feature_importance(model)
    if(shap_plot == True):
        get_shap_plot(model, X_train)
    if(decision_tree_curve == True):
        get_decision_tree_curve(X, y, tree_depth=tree_depth)

    return model


get_model(1, decision_tree_curve=True, tree_depth=3, shap_plot=True)