import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine = load_wine()
X = wine.data
y = wine.target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the params for ML Model
max_depth = 8
n_estimators = 10

mlflow.set_experiment("Exp 1: wine_quality_classification")  # to create an experiment

with mlflow.start_run(): # using experiment id 
    # define the model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)

    # fit the model
    rf.fit(X_train, y_train)

    # predict
    y_pred = rf.predict(X_test)

    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    # log metrics
    mlflow.log_metric("accuracy", accuracy)

    # log params
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # log model
    mlflow.sklearn.log_model(rf, "model")

    # log confusion matrix
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("mlflow.note.content", "This is a simple example of a classification model using Random Forest on the Wine dataset.")

    print(f"Accuracy: {accuracy}")