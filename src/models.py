
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["AGEP", "WKHP"]
CATEGORICAL_FEATURES = [
    "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "RAC1P"
]
RANDOM_STATE = 42


def create_preprocessor():
    """Build preprocessing for numerical and categorical features."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def create_logistic_regression():
    """Create a logistic regression pipeline."""
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def create_random_forest():
    """Create a random forest pipeline."""
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
