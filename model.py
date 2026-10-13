import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle


def train_model():
    """Train the model, print metrics, save to pickle, and return everything."""
    df = pd.read_csv('student_score_data.csv')
    print(df.head())
    df.info()

    data = df.drop(columns=[
        "student_id",
        "gender",
        "age",
        "past_failures",
        "internet_access",
        "paid_classes",
        "family_support",
        "school_support",
        "extra_activities",
        "wants_higher_edu",
        "free_time",
        "going_out",
        "health",
        "data_source"
    ])
    print(data.head())

    x = data.drop(columns=["final_score"])
    y = data["final_score"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("_" * 100)
    print("model performance")
    print("_" * 100)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")

    print("_" * 100)
    print("Predictions:")
    print(predictions[:10])
    print("Actual Values:")
    print(y_test[:10])
    print("_" * 100)

    pickle.dump(model, open('model.pkl', 'wb'))

    return model, mse, rmse, mae, r2, x_train, x_test, y_train, y_test, predictions, df


# Only run interactively when executed directly (not when imported)
if __name__ == "__main__":
    model, mse, rmse, mae, r2, *_ = train_model()

    sd_hour = int(input("enter study hour per week: "))
    attd_per = int(input("enter attendance percentage: "))
    per_mrk = int(input("enter previous marks: "))

    real_data = [[sd_hour, attd_per, per_mrk]]
    pre_real = model.predict(real_data)

    print("predicted value", pre_real)