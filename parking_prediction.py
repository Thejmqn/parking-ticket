import re
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import holidays
import meteostat
import pandas as pd
import requests
from meteostat import daily, Point


NCAA_API_BASE_URL = "https://ncaa-api.henrygd.me"
CHARLOTTESVILLE = Point(38.0293, -78.4767)
LOCAL_TIMEZONE = "America/New_York"


def _is_virginia_team(team):
    if not isinstance(team, dict):
        return False

    names = team.get("names", {})
    if not isinstance(names, dict):
        return False

    identifiers = {
        str(names.get(key, "")).strip().casefold()
        for key in ("seo", "short", "full")
    }
    return bool(
        identifiers
        & {"virginia", "virginia-cavaliers", "virginia cavaliers"}
    )


def _extract_virginia_game_dates(payload):
    games = payload.get("games")
    if not isinstance(games, list):
        raise ValueError("NCAA scoreboard response does not contain a games list")

    game_dates = set()
    for item in games:
        game = item.get("game", item) if isinstance(item, dict) else None
        if not isinstance(game, dict):
            continue

        if not any(
            _is_virginia_team(game.get(side))
            for side in ("home", "away")
        ):
            continue

        start_date = game.get("startDate")
        if not start_date:
            raise ValueError("Virginia game in NCAA response is missing startDate")
        game_dates.add(datetime.strptime(start_date, "%m/%d/%Y").date())

    return game_dates


def fetch_uva_game_dates(years):
    game_dates = set()
    last_request_at = None

    with requests.Session() as session:

        def get_json(path):
            nonlocal last_request_at
            if last_request_at is not None:
                delay = 0.35 - (time.monotonic() - last_request_at)
                if delay > 0:
                    time.sleep(delay)

            response = session.get(
                f"{NCAA_API_BASE_URL}{path}",
                timeout=30,
            )
            last_request_at = time.monotonic()
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"NCAA API returned an invalid response for {path}")
            return payload

        for year in years:
            print(f"  Fetching NCAA FBS schedule for {year}...")
            schedule = get_json(f"/schedule/football/fbs/{year}")
            weeks = schedule.get("gameWeeks")
            if not isinstance(weeks, list) or not weeks:
                raise ValueError(f"NCAA schedule has no game weeks for {year}")

            season_game_dates = set()
            for week_info in weeks:
                if not isinstance(week_info, dict) or not week_info.get("week"):
                    raise ValueError(f"NCAA schedule has an invalid week for {year}")
                if int(week_info.get("games", 0)) == 0:
                    continue

                week = str(week_info["week"])
                scoreboard = get_json(
                    f"/scoreboard/football/fbs/{year}/{week}/all-conf"
                )
                season_game_dates.update(_extract_virginia_game_dates(scoreboard))

            if not season_game_dates:
                raise ValueError(f"No Virginia football games found for {year}")
            game_dates.update(season_game_dates)
            print(
                f"  Found {len(season_game_dates)} Virginia game dates for {year}."
            )

    return game_dates


def _weather_features(issue_dates):
    date_values = issue_dates.dropna()
    if date_values.empty:
        return pd.DataFrame(
            index=issue_dates.index,
            columns=["IsSnowing", "IsRaining", "IsAbove100F", "IsBelow32F"],
        ).astype("boolean")

    start = min(date_values)
    end = max(date_values)
    print(f"  Fetching Meteostat daily observations from {start} through {end}...")
    nearby_stations = meteostat.stations.nearby(CHARLOTTESVILLE, limit=4)
    if nearby_stations.empty:
        raise RuntimeError("Meteostat found no weather stations near Charlottesville.")

    daily_weather = meteostat.interpolate(
        daily(
            nearby_stations,
            start,
            end,
            parameters=[
                meteostat.Parameter.PRCP,
                meteostat.Parameter.TMIN,
                meteostat.Parameter.TMAX,
            ],
        ),
        CHARLOTTESVILLE,
    ).fetch()
    if daily_weather is None:
        daily_weather = pd.DataFrame()
    if daily_weather.empty:
        daily_features = pd.DataFrame(
            columns=["IsAbove100F", "IsBelow32F"]
        )
    else:
        required_columns = {"tmax", "tmin"}
        missing_columns = required_columns.difference(daily_weather.columns)
        if missing_columns:
            raise ValueError(
                "Meteostat daily data is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        daily_weather.index = pd.to_datetime(daily_weather.index).date
        maximum_temperature = daily_weather["tmax"]
        minimum_temperature = daily_weather["tmin"]

        daily_features = pd.DataFrame(
            {
                "IsAbove100F": maximum_temperature.gt((100 - 32) * 5 / 9).mask(
                    maximum_temperature.isna()
                ),
                "IsBelow32F": minimum_temperature.lt(0).mask(
                    minimum_temperature.isna()
                ),
            },
            index=daily_weather.index,
        ).astype("boolean")

    hourly_feature_frames = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=365 * 3 - 1), end)
        print(
            "  Fetching hourly weather conditions from "
            f"{chunk_start} through {chunk_end}..."
        )
        hourly_weather = meteostat.interpolate(
            meteostat.hourly(
                nearby_stations,
                chunk_start,
                chunk_end,
                parameters=[meteostat.Parameter.COCO],
            ),
            CHARLOTTESVILLE,
        ).fetch()
        if hourly_weather is not None and not hourly_weather.empty:
            if "coco" not in hourly_weather.columns:
                raise ValueError(
                    "Meteostat hourly data is missing the coco condition code."
                )
            hourly_weather.index = pd.to_datetime(hourly_weather.index)
            conditions = hourly_weather["coco"]
            condition_codes = conditions.groupby(conditions.index.date)

            def contains_condition(codes, categories):
                if not codes.notna().any():
                    return pd.NA
                return bool(codes.isin(categories).any())

            hourly_feature_frames.append(
                pd.DataFrame(
                    {
                        "IsSnowing": condition_codes.apply(
                            contains_condition, categories=[14, 15, 16, 21, 22]
                        ),
                        "IsRaining": condition_codes.apply(
                            contains_condition,
                            categories=[
                                7,
                                8,
                                9,
                                10,
                                11,
                                17,
                                18,
                                19,
                                20,
                                25,
                                26,
                            ],
                        ),
                    }
                ).astype("boolean")
            )
        chunk_start = chunk_end + timedelta(days=1)

    if not hourly_feature_frames:
        hourly_features = pd.DataFrame(
            columns=["IsSnowing", "IsRaining"],
            index=pd.Index([], dtype=object),
        ).astype("boolean")
        print("  Meteostat returned no hourly conditions; snow/rain will be unknown.")
    else:
        hourly_features = pd.concat(hourly_feature_frames).sort_index()

    result = pd.DataFrame(index=issue_dates.index)
    for column in ("IsSnowing", "IsRaining"):
        result[column] = issue_dates.map(hourly_features[column]).astype("boolean")
    for column in ("IsAbove100F", "IsBelow32F"):
        result[column] = issue_dates.map(daily_features[column]).astype("boolean")
    missing_weather_dates = int(result["IsSnowing"].isna().sum())
    print(
        f"  Added weather flags; {missing_weather_dates} ticket dates have "
        "no weather observation."
    )
    return result


def build_parking_tickets(csv_path=None):
    if csv_path is None:
        csv_path = Path(__file__).with_name("Parking_Tickets.csv")

    print(f"Stage 1/6: Reading parking ticket data from {csv_path}...")
    tickets = pd.read_csv(csv_path)
    print(f"  Loaded {len(tickets):,} ticket records.")

    print("Stage 2/6: Parsing ticket issue dates...")
    tickets["DateIssued"] = pd.to_datetime(
        tickets["DateIssued"], utc=True, errors="coerce"
    )
    invalid_dates = int(tickets["DateIssued"].isna().sum())
    print(f"  Parsed issue dates; {invalid_dates:,} invalid dates will be excluded.")

    print("Stage 3/5: Filtering tickets by date, violation, and street...")
    streets = [
        "JEFFERSON PARK AVE",
        "OBSERVATORY AVE",
        "HARMON ST",
        "MONTEBELLO CIR",
        "VALLEY RD",
        "WASHINGTON AVE",
        "CARROLLTON TER",
        "PRICE AVE",
        "SHAMROCK RD",
        "OBSERVATORY AV",
        "BRANDON AV",
        
    ]
    street_pattern = "|".join(re.escape(street) for street in streets)

    mask = (
        tickets["DateIssued"].ge(pd.Timestamp("2015-01-01", tz="UTC"))
        & tickets["DateIssued"].lt(pd.Timestamp("2026-12-01", tz="UTC"))
        & tickets["ViolationDescription"].str.strip().isin(
            ["Permit Zone w/o Permit", "No Parking any time"]
        )
        & tickets["StreetName"].str.contains(
            street_pattern, case=False, na=False, regex=True
        )
    )
    parking_tickets = tickets.loc[mask].copy()
    print(f"  Selected {len(parking_tickets):,} matching tickets.")

    print("Stage 4/5: Deriving Charlottesville-local dates and U.S. holidays...")
    issue_dates = (
        parking_tickets["DateIssued"]
        .dt.tz_convert(LOCAL_TIMEZONE)
        .dt.date
    )
    years = sorted({issue_date.year for issue_date in issue_dates.dropna()})

    federal_holidays = holidays.country_holidays(
        "US",
        years=years,
        observed=True,
    )
    parking_tickets["IsUSHoliday"] = issue_dates.isin(federal_holidays)
    print(
        f"  Added holiday flags for {len(years)} ticket year(s); "
        f"{int(parking_tickets['IsUSHoliday'].sum()):,} tickets fall on holidays."
    )

    print("Stage 5/5: Matching tickets to University of Virginia football game dates...")
    uva_game_dates = fetch_uva_game_dates(years)
    parking_tickets["IsUVAFootballGameDay"] = issue_dates.isin(uva_game_dates)
    print(
        f"  Added game-day flags using {len(uva_game_dates)} distinct game date(s); "
        f"{int(parking_tickets['IsUVAFootballGameDay'].sum()):,} tickets fall "
        "on game days."
    )

    print("Adding Charlottesville daily weather flags...")
    weather_features = _weather_features(issue_dates)
    for column in weather_features:
        parking_tickets[column] = weather_features[column]

    print(
        f"Data ingest complete: {len(parking_tickets):,} tickets with "
        f"{len(parking_tickets.columns)} columns."
    )
    return parking_tickets


if __name__ == "__main__":
    parking_tickets = build_parking_tickets()
    output_path = Path(__file__).with_name("parking_tickets_enriched.csv")
    print(f"Saving enriched parking ticket data to {output_path}...")
    parking_tickets.to_csv(output_path, index=False)
    print(f"Saved {len(parking_tickets):,} records.")
