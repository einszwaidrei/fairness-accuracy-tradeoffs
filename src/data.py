
from pathlib import Path

import pandas as pd
from folktables import ACSDataSource, ACSIncome, BasicProblem


def load_acs_income(state="IL", year="2018"):
    """Load ACSIncome data and separate features, labels, and sex groups."""

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    data_source = ACSDataSource(
        survey_year=year,
        horizon="1-Year",
        survey="person",
        root_dir=str(data_dir),
    )

    state_data = data_source.get_data(
        states=[state],
        download=True,
    )

    income_task = BasicProblem(
        features=ACSIncome.features,
        target=ACSIncome.target,
        target_transform=ACSIncome.target_transform,
        group="SEX",
        preprocess=ACSIncome._preprocess,
        postprocess=ACSIncome._postprocess,
    )

    features, labels, sex_groups = income_task.df_to_numpy(state_data)

    X = pd.DataFrame(
        features,
        columns=ACSIncome.features,
    ).drop(columns=["SEX"])

    y = pd.Series(labels, name="income_above_50k")
    A = pd.Series(sex_groups, name="SEX")

    return X, y, A
