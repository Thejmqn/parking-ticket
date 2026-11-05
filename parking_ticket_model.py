from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
WEATHER_FEATURES = [
    "IsSnowing",
    "IsRaining",
    "IsAbove100F",
    "IsBelow32F",
]
MODEL_FEATURES = [
    "IsUSHoliday",
    "IsUVAFootballGameDay",
    *WEATHER_FEATURES,
]
WEATHER_MISSING_FEATURE = "WeatherUnknown"
PREDICTOR_FEATURES = [*MODEL_FEATURES, WEATHER_MISSING_FEATURE]
LOCAL_TIMEZONE = "America/New_York"


def _read_ticket_data(csv_path):
    tickets = pd.read_csv(csv_path, low_memory=False)
    required_columns = {"DateIssued", "TimeIssued", *MODEL_FEATURES}
    missing_columns = required_columns.difference(tickets.columns)
    if missing_columns:
        raise ValueError(
            "Enriched ticket data is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    tickets["DateIssued"] = pd.to_datetime(
        tickets["DateIssued"], utc=True, errors="coerce"
    )
    tickets = tickets.loc[tickets["DateIssued"].notna()].copy()
    if tickets.empty:
        raise ValueError("Enriched ticket data has no valid issue dates.")

    local_dates = (
        tickets["DateIssued"]
        .dt.tz_convert(LOCAL_TIMEZONE)
        .dt.date
    )
    time_parts = tickets["TimeIssued"].astype("string").str.extract(
        r"^\s*(\d{1,2}):(\d{2})\s*$"
    )
    hour = pd.to_numeric(time_parts[0], errors="coerce")
    minute = pd.to_numeric(time_parts[1], errors="coerce")
    valid_time = hour.between(0, 23) & minute.between(0, 59)
    if not valid_time.all():
        raise ValueError(
            f"{int((~valid_time).sum())} ticket rows have an invalid TimeIssued value."
        )

    tickets["LocalDate"] = local_dates
    tickets["Hour"] = hour.astype(int)
    return tickets


def _build_hourly_exposure(tickets):
    calendar_dates = np.sort(tickets["LocalDate"].unique())
    start_date = calendar_dates[0]
    end_date = calendar_dates[-1]
    print(
        "Building 24 hourly exposure intervals for each date represented in "
        f"the enriched data ({start_date} through {end_date})..."
    )

    calendar = pd.DataFrame(
        {
            "LocalDate": np.repeat(calendar_dates, 24),
            "Hour": np.tile(np.arange(24), len(calendar_dates)),
        }
    )
    calendar["Weekday"] = pd.Categorical(
        [day.strftime("%A") for day in calendar["LocalDate"]],
        categories=WEEKDAYS,
        ordered=True,
    )

    print("Reusing holiday, game-day, and weather flags from enriched ticket data...")
    observed_features = tickets.groupby("LocalDate", sort=False)[MODEL_FEATURES].agg(
        lambda values: values.dropna().iloc[0] if values.notna().any() else pd.NA
    )
    calendar = calendar.merge(
        observed_features,
        left_on="LocalDate",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    hourly_counts = tickets.groupby(["LocalDate", "Hour"], observed=True).size()
    calendar_index = pd.MultiIndex.from_frame(calendar[["LocalDate", "Hour"]])
    calendar["TicketCount"] = (
        hourly_counts.reindex(calendar_index, fill_value=0).to_numpy()
    )

    calendar[WEATHER_MISSING_FEATURE] = calendar[WEATHER_FEATURES].isna().any(axis=1)
    for feature in WEATHER_FEATURES:
        calendar[feature] = calendar[feature].astype("boolean").fillna(False).astype(bool)

    for feature in ("IsUSHoliday", "IsUVAFootballGameDay", WEATHER_MISSING_FEATURE):
        calendar[feature] = calendar[feature].astype("boolean").fillna(False).astype(bool)

    return calendar


def fit_ticket_likelihood_model(
    input_path=None,
    output_path=None,
):
    script_directory = Path(__file__).resolve().parent
    input_path = input_path or script_directory / "parking_tickets_enriched.csv"
    output_path = output_path or script_directory / "parking_ticket_likelihood_by_hour.csv"

    print(f"Loading enriched ticket data from {input_path}...")
    tickets = _read_ticket_data(input_path)
    print(f"Loaded {len(tickets):,} ticket records.")

    exposure = _build_hourly_exposure(tickets)
    active_features = [
        feature
        for feature in PREDICTOR_FEATURES
        if exposure[feature].nunique(dropna=False) > 1
    ]
    excluded_features = sorted(set(PREDICTOR_FEATURES) - set(active_features))
    if excluded_features:
        print(
            "  Excluding constant predictors with no variation in this dataset: "
            + ", ".join(excluded_features)
        )
    formula = "TicketCount ~ C(Hour) + C(Weekday)"
    if active_features:
        formula += " + " + " + ".join(
            f"C({feature})" for feature in active_features
        )

    print("Estimating negative-binomial overdispersion...")
    poisson_fit = smf.glm(
        formula=formula,
        data=exposure,
        family=sm.families.Poisson(),
    ).fit()
    fitted_counts = np.maximum(poisson_fit.fittedvalues.to_numpy(), 1e-12)
    observed_counts = exposure["TicketCount"].to_numpy()
    alpha = (
        np.sum((observed_counts - fitted_counts) ** 2 - fitted_counts)
        / np.sum(fitted_counts**2)
    )
    alpha = max(float(alpha), 1e-8)
    print(f"Estimated NB2 dispersion alpha: {alpha:.6g}")

    print("Fitting negative-binomial GLM...")
    model = smf.glm(
        formula=formula,
        data=exposure,
        family=sm.families.NegativeBinomial(alpha=alpha),
    ).fit()
    exposure["PredictedTicketCount"] = model.predict(exposure)
    exposure["PredictedTicketLikelihood"] = 1 - np.power(
        1 + alpha * exposure["PredictedTicketCount"],
        -1 / alpha,
    )

    result_columns = [
        "LocalDate",
        "Hour",
        "Weekday",
        "TicketCount",
        *MODEL_FEATURES,
        WEATHER_MISSING_FEATURE,
        "PredictedTicketCount",
        "PredictedTicketLikelihood",
    ]
    predictions = exposure[result_columns].copy()
    predictions["PredictedTicketLikelihood"] *= 100
    predictions.rename(
        columns={
            "PredictedTicketLikelihood": "PredictedTicketLikelihoodPercent"
        },
        inplace=True,
    )
    predictions.to_csv(output_path, index=False)

    print(
        f"Saved {len(predictions):,} hourly predictions to {output_path}."
    )
    print(
        "Mean in-sample probability of at least one ticket in a modeled hour: "
        f"{predictions['PredictedTicketLikelihoodPercent'].mean():.3f}%"
    )
    print(
        "Interpretation: predictions estimate ticket counts per hour on dates "
        "that appear in the enriched CSV; they are not an individual's "
        "personalized risk and do not estimate ticket-free dates."
    )
    print(
        "Missing weather values are imputed as false for individual weather "
        "flags and represented separately by WeatherUnknown."
    )
    return model, predictions


def fit_grouped_ticket_likelihood_model(
    input_path=None,
    output_path=None,
):
    script_directory = Path(__file__).resolve().parent
    input_path = input_path or script_directory / "parking_tickets_enriched.csv"
    output_path = output_path or script_directory / "parking_ticket_likelihood_grouped.csv"

    print(f"Loading enriched ticket data from {input_path}...")
    tickets = _read_ticket_data(input_path)
    print(f"Loaded {len(tickets):,} ticket records.")

    hourly_exposure = _build_hourly_exposure(tickets)
    active_features = [
        feature
        for feature in PREDICTOR_FEATURES
        if hourly_exposure[feature].nunique(dropna=False) > 1
    ]
    excluded_features = sorted(set(PREDICTOR_FEATURES) - set(active_features))
    if excluded_features:
        print(
            "  Excluding constant predictors with no variation in this dataset: "
            + ", ".join(excluded_features)
        )

    formula = "TicketCount ~ C(Hour) + C(Weekday)"
    if active_features:
        formula += " + " + " + ".join(
            f"C({feature})" for feature in active_features
        )

    print("Estimating negative-binomial dispersion from date-hour ticket counts...")
    poisson_fit = smf.glm(
        formula=formula,
        data=hourly_exposure,
        family=sm.families.Poisson(),
    ).fit()
    fitted_counts = np.maximum(poisson_fit.fittedvalues.to_numpy(), 1e-12)
    observed_counts = hourly_exposure["TicketCount"].to_numpy()
    alpha = (
        np.sum((observed_counts - fitted_counts) ** 2 - fitted_counts)
        / np.sum(fitted_counts**2)
    )
    alpha = max(float(alpha), 1e-8)
    print(f"Estimated NB2 dispersion alpha: {alpha:.6g}")

    group_columns = ["Hour", "Weekday", *PREDICTOR_FEATURES]
    grouped = (
        hourly_exposure.groupby(
            group_columns,
            observed=True,
            as_index=False,
        )
        .agg(
            TicketCount=("TicketCount", "sum"),
            ExposureHours=("TicketCount", "size"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    print(
        f"Grouped {len(hourly_exposure):,} date-hour rows into "
        f"{len(grouped):,} date-free predictor combinations."
    )

    grouped["LogExposure"] = np.log(grouped["ExposureHours"])
    grouped_formula = "TicketCount ~ C(Hour) + C(Weekday)"
    if active_features:
        grouped_formula += " + " + " + ".join(
            f"C({feature})" for feature in active_features
        )

    print("Fitting negative-binomial GLM with grouped exposure offsets...")
    model = smf.glm(
        formula=grouped_formula,
        data=grouped,
        family=sm.families.NegativeBinomial(alpha=alpha),
        offset=grouped["LogExposure"],
    ).fit()
    grouped["PredictedGroupTicketCount"] = model.predict(
        grouped,
        offset=grouped["LogExposure"],
    )
    grouped["PredictedTicketCountPerHour"] = (
        grouped["PredictedGroupTicketCount"] / grouped["ExposureHours"]
    )
    grouped["PredictedTicketLikelihoodPercent"] = (
        1
        - np.power(
            1 + alpha * grouped["PredictedTicketCountPerHour"],
            -1 / alpha,
        )
    ) * 100

    output_columns = [
        *group_columns,
        "ExposureHours",
        "TicketCount",
        "PredictedGroupTicketCount",
        "PredictedTicketCountPerHour",
        "PredictedTicketLikelihoodPercent",
    ]
    predictions = grouped[output_columns].copy()
    predictions.to_csv(output_path, index=False)

    print(
        f"Saved {len(predictions):,} date-free grouped predictions to "
        f"{output_path}."
    )
    print(
        "Mean modeled probability of at least one ticket in an hour: "
        f"{predictions['PredictedTicketLikelihoodPercent'].mean():.3f}%"
    )
    return model, predictions


if __name__ == "__main__":
    fit_grouped_ticket_likelihood_model()
