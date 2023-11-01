# LogitBoost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.datasets import load_breast_cancer

# Загрузите датасет "Breast Cancer Wisconsin (Diagnostic)"
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте объект классификатора AdaBoost с базовым классификатором DecisionTreeClassifier
base_classifier = DecisionTreeClassifier(max_depth=1)
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, learning_rate=1)

# Обучите модель
adaboost_classifier.fit(X_train, y_train)

# Сделайте прогнозы на тестовом наборе данных
y_pred = adaboost_classifier.predict(X_test)

# Оцените качество модели
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
