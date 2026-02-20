import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import joblib

# STEP 1: Load dataset

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully")

print("Columns:", data.columns)

print("Shape:", data.shape)


# STEP 2: Split features and target

X = data.drop("y", axis=1)

y = data["y"]

print("Features shape:", X.shape)

print("Target shape:", y.shape)


# STEP 3: Train-test split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)

print("Training samples:", X_train.shape)

print("Testing samples:", X_test.shape)


# STEP 4: Create pipeline

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", RandomForestClassifier(random_state=42))

])

print("Pipeline created successfully")
# STEP 5: Define hyperparameters

param_grid = {

    "model__n_estimators": [100, 200],

    "model__max_depth": [10, 20, None],

    "model__min_samples_split": [2, 5]

}


print("Starting hyperparameter tuning...")


# STEP 6: GridSearchCV

grid_search = GridSearchCV(

    pipeline,

    param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2

)


# STEP 7: Train model

grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)


best_model = grid_search.best_estimator_


# STEP 8: Test accuracy

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)


# STEP 9: Save model

joblib.dump(best_model, "model.pkl")

print("Model saved as model.pkl")