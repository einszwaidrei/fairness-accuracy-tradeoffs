
from sklearn.model_selection import train_test_split


def prepare_data_splits(X, y, A, random_state=42):
    """Split data into 60% train, 20% validation, and 20% test."""
    X_train, X_temp, y_train, y_temp, A_train, A_temp = train_test_split(
        X, y, A,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test, A_val, A_test = train_test_split(
        X_temp, y_temp, A_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        A_train, A_val, A_test,
    )
