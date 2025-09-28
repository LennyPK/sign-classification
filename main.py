"""main.py"""

import gc

import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

import data


def main():
    """Main function to load and process traffic sign images."""

    label_map = data.load_labels()

    # For the purpose of printing, convert map to df
    label_map_df = pd.DataFrame(label_map.items(), columns=["ClassId", "Name"])
    print(label_map_df.head(10))

    flat_data, target = data.load_data(label_map)

    gc.collect()

    df = pd.DataFrame(flat_data)
    df["Target"] = target
    print(df.head())
    print(df.shape)

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y
    )
    print("Data split successfully")
    print("====== Train Data Shapes ======")
    print("X_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("====== Test Data Shapes =======")
    print("X_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print("===============================")

    # model = MLPClassifier(
    #     random_state=0, learning_rate_init=0.0001, max_iter=100000, verbose=True
    # )
    # print(model)
    # print("MLP Model configured")
    # model.fit(x_train, y_train)
    # print("MLP Model fitted")
    # y_pred = model.predict(x_test)

    # acc_score = accuracy_score(y_test, y_pred)
    # print(f"MLP Accuracy Score: {acc_score * 100:.2f}%") # 97.74%

    # SVC model
    # param_grid = {
    #     "C": [0.1, 1, 10],
    #     "gamma": [0.0001, 0.001, 0.1],
    #     "kernel": ["linear", "rbf"],
    # }
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["poly"],
    }

    svc = svm.SVC(verbose=True)
    print("The training of the model has started")
    model = GridSearchCV(svc, param_grid, verbose=2, n_jobs=-1, cv=2)
    print(model)
    print("The model is being fitted")
    model.fit(x_train, y_train)
    print("The training of the model has ended")
    print("Best parameters found: ", model.best_params_)

    y_pred_svc = model.predict(x_test)
    acc_score_svc = accuracy_score(y_test, y_pred_svc)
    print(f"SVC Accuracy Score: {acc_score_svc * 100:.2f}%")

    """Poly SVC with best parameters from GridSearchCV"""
    # Parameter grid: "C": [0.1, 1, 10], "kernel": ["poly"]
    # Best parameters found: {'C': 10, 'kernel': 'poly'}
    # SVC Accuracy Score: 84.82%

    """RBF SVC with best parameters from GridSearchCV"""
    # Parameter grid: "C": [0.1, 1, 10], "gamma": [0.0001, 0.001, 0.1], "kernel": ["rbf"]
    # Best parameters found: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
    # SVC Accuracy Score: 95.08%

    """Linear SVC with best parameters from GridSearchCV"""
    # Parameter grid: "C": [0.1, 1, 10], "kernel": ["linear"]
    # Best parameters found: {'C': 1, 'kernel': 'linear'}
    # SVC Accuracy Score: 98.14%

    """Linear SVC with custom parameters (1 fold)"""
    # clf = svm.SVC(kernel="linear", C=1, gamma="scale", verbose=True)
    # print(clf)
    # print("SVC Model configured")
    # clf.fit(x_train, y_train)
    # print("SVC Model fitted")
    # y_pred_svc = clf.predict(x_test)
    # acc_score_svc = accuracy_score(y_test, y_pred_svc)
    # print(f"SVC Accuracy Score: {acc_score_svc * 100:.2f}%")  # 98.14%

    """RBF SVC with custom parameters (1 fold)"""
    # clf = svm.SVC(kernel="rbf", C=1, gamma=0.001, verbose=True)
    # print(clf)
    # print("SVC Model configured")
    # clf.fit(x_train, y_train)
    # print("SVC Model fitted")
    # y_pred_svc = clf.predict(x_test)
    # acc_score_svc = accuracy_score(y_test, y_pred_svc)
    # print(f"SVC Accuracy Score: {acc_score_svc * 100:.2f}%") #80.50%


if __name__ == "__main__":
    main()
