from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FACTOR_COMPARISONS = [
    ("IsUSHoliday", "Non-holiday", "Holiday", "Holiday"),
    (
        "IsUVAFootballGameDay",
        "Non-game day",
        "Game day",
        "UVA football game",
    ),
    ("IsSnowing", "Not snowing", "Snowing", "Snow"),
    ("IsRaining", "Not raining", "Raining", "Rain"),
    ("IsAbove100F", "Not above 100°F", "Above 100°F", "Above 100°F"),
    ("IsBelow32F", "Not below 32°F", "Below 32°F", "Below 32°F"),
]
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _as_boolean(values):
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.map({"true": True, "false": False})


def _weighted_likelihood(data, feature, unknown_weather):
    subset = data
    if unknown_weather and "WeatherUnknown" in data.columns:
        subset = data.loc[~_as_boolean(data["WeatherUnknown"]).fillna(True)]

    flags = _as_boolean(subset[feature])
    subset = subset.loc[flags.notna()].copy()
    subset[feature] = flags.loc[subset.index]
    if subset.empty:
        return pd.DataFrame(columns=["label", "likelihood", "exposure"])

    grouped = (
        subset.groupby(feature, observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "likelihood": (
                        group["PredictedTicketLikelihoodPercent"]
                        * group["ExposureHours"]
                    ).sum()
                    / group["ExposureHours"].sum(),
                    "exposure": group["ExposureHours"].sum(),
                }
            ),
            include_groups=False,
        )
    )
    labels = {False: "No", True: "Yes"}
    grouped["label"] = [labels[value] for value in grouped.index]
    return grouped.reset_index(drop=True)


def plot_grouped_ticket_likelihood(input_path=None, output_path=None):
    script_directory = Path(__file__).resolve().parent
    input_path = input_path or script_directory / "parking_ticket_likelihood_grouped.csv"
    output_path = output_path or script_directory / "parking_ticket_likelihood_factors.png"

    print(f"Reading grouped predictions from {input_path}...")
    data = pd.read_csv(input_path, low_memory=False)
    required = {
        "ExposureHours",
        "PredictedTicketLikelihoodPercent",
        *(feature for feature, *_ in FACTOR_COMPARISONS),
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Grouped predictions are missing columns: " + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError("Grouped predictions contain no rows.")

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    figure.suptitle(
        "Average Predicted Parking-Ticket Likelihood by Factor",
        fontsize=16,
    )
    colors = ["#4C78A8", "#F58518"]

    for axis, (feature, false_label, true_label, title) in zip(
        axes.flat, FACTOR_COMPARISONS
    ):
        comparison = _weighted_likelihood(
            data,
            feature,
            unknown_weather=feature
            in {"IsSnowing", "IsRaining", "IsAbove100F", "IsBelow32F"},
        )
        axis.set_title(title)
        if comparison.empty:
            axis.text(
                0.5,
                0.5,
                "No observations",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            continue

        wanted_labels = ["No", "Yes"]
        comparison = comparison.set_index("label").reindex(wanted_labels)
        present = comparison["likelihood"].notna()
        if not present.any():
            axis.text(
                0.5,
                0.5,
                "No observations",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            continue

        labels = [
            false_label if label == "No" else true_label
            for label in comparison.index[present]
        ]
        values = comparison.loc[present, "likelihood"]
        exposures = comparison.loc[present, "exposure"]
        bar_colors = [colors[0] if label == "No" else colors[1]
                      for label in comparison.index[present]]
        bars = axis.bar(labels, values, color=bar_colors)
        axis.bar_label(
            bars,
            labels=[
                f"{value:.2f}%\n({int(exposure):,} hours)"
                for value, exposure in zip(values, exposures)
            ],
            padding=3,
            fontsize=8,
        )
        if not present.all():
            missing_label = true_label if not present.loc["Yes"] else false_label
            axis.text(
                0.5,
                0.92,
                f"No {missing_label.lower()} observations",
                ha="center",
                va="top",
                transform=axis.transAxes,
                fontsize=8,
            )
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=0.25)

    for axis in axes[:, 0]:
        axis.set_ylabel("Exposure-weighted predicted likelihood (%)")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved factor comparison bar graphs to {output_path}.")
    return output_path


def plot_hour_of_week_likelihood(input_path=None, output_path=None):
    script_directory = Path(__file__).resolve().parent
    input_path = input_path or script_directory / "parking_ticket_likelihood_grouped.csv"
    output_path = output_path or script_directory / "parking_ticket_likelihood_by_hour_of_week.png"

    print(f"Reading grouped predictions from {input_path}...")
    data = pd.read_csv(input_path, low_memory=False)
    required = {
        "Hour",
        "Weekday",
        "ExposureHours",
        "PredictedTicketLikelihoodPercent",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Grouped predictions are missing columns: " + ", ".join(sorted(missing))
        )
    if data.empty:
        raise ValueError("Grouped predictions contain no rows.")

    invalid_hours = ~data["Hour"].between(0, 23)
    invalid_weekdays = ~data["Weekday"].isin(WEEKDAYS)
    if invalid_hours.any() or invalid_weekdays.any():
        raise ValueError("Grouped predictions contain an invalid hour or weekday.")
    if data["ExposureHours"].isna().any() or (data["ExposureHours"] <= 0).any():
        raise ValueError("ExposureHours must contain positive values.")

    data["HourOfWeek"] = (
        data["Weekday"].map({day: index for index, day in enumerate(WEEKDAYS)}) * 24
        + data["Hour"]
    )
    data["WeightedLikelihood"] = (
        data["PredictedTicketLikelihoodPercent"] * data["ExposureHours"]
    )
    hourly = (
        data.groupby("HourOfWeek", as_index=False)
        .agg(
            WeightedLikelihood=("WeightedLikelihood", "sum"),
            ExposureHours=("ExposureHours", "sum"),
        )
    )
    hourly["PredictedTicketLikelihoodPercent"] = (
        hourly["WeightedLikelihood"] / hourly["ExposureHours"]
    )

    full_week = pd.DataFrame({"HourOfWeek": range(168)}).merge(
        hourly[["HourOfWeek", "PredictedTicketLikelihoodPercent"]],
        on="HourOfWeek",
        how="left",
        validate="one_to_one",
    )
    figure, axis = plt.subplots(figsize=(16, 6))
    axis.plot(
        full_week["HourOfWeek"],
        full_week["PredictedTicketLikelihoodPercent"],
        color="#4C78A8",
        linewidth=1.8,
    )
    axis.set_xlim(0, 167)
    axis.set_xticks(range(0, 168, 12))
    axis.set_xticklabels(
        [
            f"{WEEKDAYS[hour_of_week // 24]} "
            f"{(hour_of_week % 24) % 12 or 12} "
            f"{'AM' if hour_of_week % 24 < 12 else 'PM'}"
            for hour_of_week in range(0, 168, 12)
        ],
        rotation=35,
        ha="right",
    )
    axis.set_xlabel("Hour of week (local weekday and time)")
    axis.set_ylabel("Exposure-weighted predicted ticket likelihood (%)")
    axis.set_title("Predicted Parking-Ticket Likelihood by Hour of Week")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(
        f"Saved hour-of-week line chart to {output_path} "
        f"({hourly['HourOfWeek'].nunique()} observed hour scores)."
    )
    return output_path


if __name__ == "__main__":
    plot_grouped_ticket_likelihood()
    plot_hour_of_week_likelihood()
