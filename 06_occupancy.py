# ruff: noqa: E402
# %%
from pathlib import Path
from typing import TypedDict

from traffic.core import Airspace, Flight, Traffic

# %%

figure_path = Path("../../../paper/c43_contrail_or_not/figures")

data_path = Path("./data")

# %%
t = Traffic.from_file(
    data_path / "eu_flights_2022feb20_filter_resample_meteo.parquet.gz"
)

t_opt = Traffic.from_file(data_path / "all_optimized_resampled.parquet.gz")

t03 = t_opt.query("not flight_id.str.endswith('_06')")
t06 = t_opt.query("not flight_id.str.endswith('_03')")


# %%
import pandas as pd
from traffic.data import aixm_airspaces

# %%
# useful to search for the ACC of a given zone
subset_area = aixm_airspaces.data.query(
    '(type == "SECTOR_C" | type == "SECTOR") & '
    'designator.str.startswith("LR") & name.notnull()'
).drop_duplicates("identifier")  # .data.designator.unique()

pd.DataFrame.from_records(
    {
        "designator": row["designator"],
        "area": aixm_airspaces[row["designator"]].area,
        "name": row["name"],
        "upper": aixm_airspaces[row["designator"]].elements[-1].upper,
    }
    for _, row in subset_area.iterrows()
).sort_values("area").tail(30)

# %%
from traffic.core import Airspace

acc_table = (
    pd.read_table("data/europe_acc.txt", sep="|", header=0, skiprows=[1])
    .rename(columns=str.strip)
    .eval("Designator = Designator.str.strip()")
    .eval("Name = Name.str.strip()")[["Designator", "Name"]]
    .dropna()
    .query('Designator != ""')
    .groupby(["Name"])
    .agg({"Designator": list})
    .reset_index()
)
airspaces: dict[str, Airspace] = {}
airspace: Airspace

for row, entry in acc_table.iterrows():
    airspace = sum(aixm_airspaces[d] for d in entry["Designator"])  # type: ignore
    airspaces[entry["Name"]] = airspace

# %%
from ipyleaflet import Map

m = Map(center=(43, 10), zoom=5)

for airspace in airspaces.values():
    m.add(airspace.above(200))

m


# %%


class Entry(TypedDict):
    flight_id: None | str
    airspace: str
    start: None | pd.Timestamp
    stop: None | pd.Timestamp


def occupancy_airspace(flight: Flight, airspace: Airspace, name: str) -> None | Entry:
    start = None
    stop = None
    for element in airspace.elements:
        lower = element.lower
        upper = element.upper
        if alt := flight.query(f"{lower} <= altitude & altitude < {upper}"):
            if alt.intersects(element.polygon):
                clip = alt.clip(airspace)
                if clip is None:
                    continue
                if start is None or start > clip.start:
                    start = clip.start
                if stop is None or stop > clip.stop:
                    stop = clip.stop

    if start is not None and stop is not None:
        return {
            "flight_id": flight.flight_id,  # type: ignore
            "airspace": name,
            "start": start,
            "stop": stop,
        }
    return None


def occupancy_flight(flight: Flight) -> None | pd.DataFrame:
    entries = [
        entry
        for (name, airspace) in airspaces.items()
        if (entry := occupancy_airspace(flight, airspace, name))
    ]
    if len(entries) > 0:
        return pd.DataFrame.from_records(entries)
    return None


df = (
    t.iterate_lazy()
    .pipe(occupancy_flight)
    .eval(desc="", max_workers=24, cache_file="data/cache/occupancy_original.parquet")
)

df03 = (
    t03.iterate_lazy()
    .pipe(occupancy_flight)
    .eval(desc="", max_workers=24, cache_file="data/cache/occupancy_03.parquet")
)

df06 = (
    t06.iterate_lazy()
    .pipe(occupancy_flight)
    .eval(desc="", max_workers=24, cache_file="data/cache/occupancy_06.parquet")
)

# %%
line = pd.concat(
    [
        df.data.eval("""
            start = start.dt.round('10min')
            stop = stop.dt.round('10min')
        """)
        .query("start <= @line & @line <= stop")
        .groupby(["airspace"])
        .agg({"flight_id": "nunique"})
        .assign(line=line)
        for line in pd.date_range("2022-02-20", "2022-02-21", freq="10 min", tz="utc")
    ]
)


# %%
import altair as alt

step = 20
overlap = 1
alt.data_transformers.disable_max_rows()
chart = (
    alt.Chart(line.reset_index(), height=step, width=600)
    .mark_area(interpolate="monotone", fillOpacity=0.8)
    .encode(
        alt.Y("flight_id").axis(None).scale(range=[step, -step * overlap]),
        alt.X("line").title(None).axis(labelFont="Roboto Condensed", labelFontSize=14),
        alt.Color("airspace").legend(None),
    )
    .transform_filter(
        "datum.airspace != 'Reykjavik ACC' & "
        "datum.airspace != 'Chisinau ACC' & "
        "datum.airspace != 'Shanwick Oceanic OCA'"
    )
    .facet(
        row=alt.Row("airspace")
        .title(None)
        .header(
            labelAngle=0,
            labelAlign="left",
            labelFont="Roboto Condensed",
            labelFontSize=16,
        )
    )
    .properties(bounds="flush")
    .configure_facet(spacing=0)
)
# chart.save("occupancy.pdf")
chart

# %%
df03 = pd.read_parquet("data/occupancy_03.parquet")
df03.flight_id.str.split("_").str[-1].unique()

# %%
df06 = pd.read_parquet("data/occupancy_06.parquet")
df06.flight_id.str.split("_").str[-1].unique()

# %%
df03_arpege = df03.query('flight_id.str.endswith("arpege_03")')
df03_fuel = df03.query('flight_id.str.endswith("fuel")')
df03_era5 = df03.query('flight_id.str.endswith("era5_03")')
df03_original = df03.query(
    'flight_id.str.endswith("1") or flight_id.str.endswith("2") or flight_id.str.endswith("4") or flight_id.str.endswith("0")'
)

df06_arpege = df06.query('flight_id.str.endswith("arpege_06")')
df06_fuel = df06.query('flight_id.str.endswith("fuel")')
df06_era5 = df06.query('flight_id.str.endswith("era5_06")')
df06_original = df06.query(
    'flight_id.str.endswith("1") or flight_id.str.endswith("2") or flight_id.str.endswith("4") or flight_id.str.endswith("0")'
)


# %%
def line_f(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            df.eval("""
            start = start.dt.round('10min')
            stop = stop.dt.round('10min')
        """)
            .query("start <= @line & @line <= stop")
            .groupby(["airspace"])
            .agg({"flight_id": "nunique"})
            .assign(line=line)
            .reset_index()
            for line in pd.date_range(
                "2022-02-20", "2022-02-21", freq="10 min", tz="utc"
            )
        ]
    )


line03_arpege = line_f(df03_arpege)
line03_era5 = line_f(df03_era5)
line03_fuel = line_f(df03_fuel)
line03_original = line_f(df03_original)
line06_arpege = line_f(df06_arpege)
line06_era5 = line_f(df06_era5)
line06_fuel = line_f(df06_fuel)
line06_original = line_f(df06_original)

# %%
diff = line03_arpege.set_index(["airspace", "line"]) - line03_original.set_index(
    ["airspace", "line"]
)

# %%
diff = line06_arpege.set_index(["airspace", "line"]) - line06_original.set_index(
    ["airspace", "line"]
)

# %%
diff = line06_fuel.set_index(["airspace", "line"]) - line06_original.set_index(
    ["airspace", "line"]
)

# %%
line = diff.reset_index()

# %%
diff.reset_index().groupby("airspace").agg(
    {"flight_id": ["min", "max", "median", "std"]}
).sort_values(("flight_id", "std"), ascending=False)

# %%

import altair as alt

alt.data_transformers.disable_max_rows()


def make_chart(diff: pd.DataFrame, title: str) -> alt.FacetChart:
    step = 20
    overlap = 1
    base = alt.Chart(diff.reset_index().assign(zero=0), height=step, width=400)

    chart: alt.FacetChart = (
        (
            base.mark_bar(binSpacing=3).encode(
                alt.X("flight_id")
                .bin(alt.Bin(extent=(-30, 30), maxbins=30))
                .title("More flights after optimisation »")
                .axis(
                    titleAnchor="end",
                    labelFont="Roboto Condensed",
                    labelFontSize=16,
                    titleFont="Roboto Condensed",
                    titleFontSize=16,
                    titleFontWeight="normal",
                ),
                alt.Y("count()")
                .title(None)
                .axis(None)
                .scale(range=[step, -step * overlap]),
                alt.Color("airspace")
                # .sort(field="flight_id", op="stdev", order="descending")
                .legend(None),
            )
            + base.mark_rule(color="black", size=3).encode(x="mean(zero):Q")
        )
        .facet(
            row=alt.Row("airspace")
            .sort(field="flight_id", op="mean", order="descending")
            .title(None)
            .header(
                labelAngle=0,
                labelAlign="left",
                labelFont="Roboto Condensed",
                labelFontSize=16,
            ),
        )
        .transform_filter(
            alt.FieldOneOfPredicate(
                "airspace",
                [
                    "Brest ACC",
                    "Reims ACC",
                    "Madrid ACC",
                    "Paris ACC",
                    "Lisboa ACC",
                    "London ACC",
                    # "Ankara ACC",
                    "Bordeaux ACC",
                ],
            )
        )
        .properties(bounds="flush", title=title)
        .configure_title(fontSize=16, font="Roboto Condensed", align="center")
        .configure_facet(spacing=0)
    )
    return chart


# %%
chart = make_chart(
    line06_fuel.set_index(["airspace", "line"])
    - line06_original.set_index(["airspace", "line"]),
    "Fuel-optimal trajectories",
)
chart.save(figure_path / "occupancy_fuel_optimal.pdf")
chart

# %%
chart = make_chart(
    line03_arpege.set_index(["airspace", "line"])
    - line03_original.set_index(["airspace", "line"]),
    "Contrail-optimal (c=0.3) trajectories",
)
chart.save(figure_path / "occupancy_contrail_03.pdf")
chart
# %%

chart = make_chart(
    line06_arpege.set_index(["airspace", "line"])
    - line06_original.set_index(["airspace", "line"]),
    "Contrail-optimal (c=0.6) trajectories",
)
chart.save(figure_path / "occupancy_contrail_06.pdf")
chart

# %%
