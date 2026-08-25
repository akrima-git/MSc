import pandas as pd
from sklearn.model_selection import KFold

def load_raw_data(path="data.csv"):
    df = pd.read_csv(path)

    return df

def get_cv_folds(df, n_splits=10, random_state=42):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(df))

def add_vehicle_age(df):
    # Convert both date columns from string/object to actual datetime objects
    # so we can do date arithmetic on them (subtraction wouldn't work on plain strings)
    df["training_date"] = pd.to_datetime(df["training_date"])
    df["year_plate_start_date"] = pd.to_datetime(df["year_plate_start_date"])
    
    # Subtract the two dates to get a Timedelta (a duration), then extract the
    # number of days as a plain integer using .dt.days
    df["vehicle_age_days"] = (df["training_date"] - df["year_plate_start_date"]).dt.days
    
    # Also express age in years (days / 365.25 to roughly account for leap years)
    df["vehicle_age_years"] = df["vehicle_age_days"] / 365.25
    
    return df

def load_clean_data(path="data.csv"):

    df = load_raw_data(path)

    df.columns = df.columns.str.strip().str.lower()

    # Derive vehicle age from the date columns (see function above)
    df = add_vehicle_age(df)


    # Unneeded columns for model training - at_derivative_id has extremely high cardinality (not feasible for processing)
    df = df.drop(columns=[
        "training_date",
        "year_plate_start_date",
        "id",
        "at_derivative_id"
    ])

    # 1 missing price, drop listing as no practical impact
    df = df.dropna(subset=["list_price"])

    df['engine_missing'] = df["engine"].isnull().astype(int)  # 1 = was missing, 0 = was present
    df["engine"] = df["engine"].fillna(0)

    df['trim'] = df['trim'].fillna('Missing') # Fill missing trim with "Missing"

    lower_bound = df['advertised_price'].quantile(0.01)  # 1st percentile
    upper_bound = df['advertised_price'].quantile(0.99)  # 99th percentile

    # Keep only rows within this range, i.e. exclude the bottom and top 1%
    df = df[
        (df['advertised_price'] >= lower_bound) & 
        (df['advertised_price'] <= upper_bound)
    ]

    df = df.reset_index(drop=True)

    return df