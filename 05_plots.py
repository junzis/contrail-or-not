# %%
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from cartopy.feature import BORDERS
from fastmeteo.source import ArcoEra5
from openap import aero, contrail
from tqdm import tqdm
from traffic.core import Flight, Traffic

pd.options.display.max_columns = 100

# %%
# Set default font and plot styles
matplotlib.rc("font", size=14)
matplotlib.rc("font", family="Fira Sans")
matplotlib.rc("lines", linewidth=2, markersize=8)
matplotlib.rc("grid", color="darkgray", linestyle=":")
matplotlib.rc("legend", loc="best", fontsize=13)

# %%
LAT0, LAT1 = 26, 66
LON0, LON1 = -15, 40

paper_figure_path = "../../../paper/c43_contrail_or_not/figures"

# %%
arco_grid = ArcoEra5(local_store="data/era5-zarr/", model_levels=137)


# %%
def agg_contrail_conditions(flight):
    flight = flight.assign(
        rhi=lambda d: contrail.relative_humidity(
            d.specific_humidity,
            aero.pressure(d.altitude * aero.ft),
            d.temperature,
            to="ice",
        ),
        crit_temp=lambda d: contrail.critical_temperature_water(
            aero.pressure(d.altitude * aero.ft)
        ),
        sac=lambda d: d.temperature < d.crit_temp,
        issr=lambda d: d.rhi > 1,
        persistent=lambda d: d.sac & d.issr,
    )
    return flight


# %%
print("Loading data")

t_base = Traffic.from_file("data/eu_flights_2022feb20_filter_resample_meteo.parquet.gz")

t_opt = Traffic.from_file("data/all_optimized_resampled.parquet.gz")

file_era5_cost = "data/grid_era5_smoothed.parquet.gz"
df_cost_era5 = pd.read_parquet(file_era5_cost)

file_arpege_cost = "data/grid_arpege_smoothed.parquet.gz"
df_cost_arpege = pd.read_parquet(file_arpege_cost).fillna(0)

# %%

ncols = 2
nrows = 3
levels = ncols * nrows
select_hour = 10
df_cost = df_cost_arpege
param = "cost"


proj = ccrs.TransverseMercator(central_longitude=10, central_latitude=50)

fig, axes = plt.subplots(
    nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), subplot_kw=dict(projection=proj)
)

heights = df_cost_era5.height.unique()
skip_levels = 2

for i, ax in enumerate(axes.flatten()):
    df_cost_pivot = df_cost.query(
        f"ts=={3600 * select_hour} and height=={heights[skip_levels + i]}"
    ).pivot(index="latitude", columns="longitude", values=param)

    lat, lon, val = (
        df_cost_pivot.index.values,
        df_cost_pivot.columns.values,
        df_cost_pivot.values,
    )

    ax.set_extent([-20, 40, 28, 65])
    ax.add_feature(BORDERS, lw=0.5, edgecolor="gray")
    ax.coastlines(resolution="110m", lw=0.5, color="gray")
    ax.gridlines(
        draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    ax.contourf(
        lon,
        lat,
        val,
        cmap="RdPu",
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        vmin=-df_cost[param].quantile(0.01),
        vmax=df_cost[param].quantile(0.99),
    )

    fl = heights[skip_levels + i] / aero.ft // 1000 * 10

    ax.text(0.03, 0.9, f"FL{int(fl)}", transform=ax.transAxes, fontsize=20)

plt.tight_layout()


# %%
def plot_costs_grid(df_cost: pd.DataFrame, cmap, levels=4, select_hour=10):
    proj = ccrs.TransverseMercator(central_longitude=10, central_latitude=50)

    fig, axes = plt.subplots(
        levels, 1, figsize=(5, levels * 3), subplot_kw=dict(projection=proj)
    )

    heights = df_cost.height.unique()
    skip_levels = 3

    for i, ax in enumerate(axes.flatten()):
        df_cost_pivot = df_cost.query(
            f"ts=={3600 * select_hour} and height=={heights[skip_levels + i]}"
        ).pivot(index="latitude", columns="longitude", values="cost")

        lat, lon, val = (
            df_cost_pivot.index.values,
            df_cost_pivot.columns.values,
            df_cost_pivot.values,
        )

        ax.set_extent([-20, 40, 28, 65])
        ax.add_feature(BORDERS, lw=0.5, edgecolor="gray")
        ax.coastlines(resolution="110m", lw=0.5, color="gray")
        ax.gridlines(
            draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )
        ax.contourf(
            lon,
            lat,
            val,
            cmap=cmap,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            vmin=-df_cost.cost.quantile(0.01),
            vmax=df_cost.cost.quantile(0.99),
        )

        fl = heights[skip_levels + i] / aero.ft // 1000 * 10

        ax.text(0.03, 0.9, f"FL{int(fl)}", transform=ax.transAxes, fontsize=20)

    plt.tight_layout()
    return plt


# %%
plot_costs_grid(df_cost_era5, select_hour=6, levels=4, cmap="Reds")
# plt.savefig(f"{paper_figure_path}/era5_cost_grid.png", dpi=150, bbox_inches="tight")

plot_costs_grid(df_cost_arpege, select_hour=6, levels=4, cmap="RdPu")
# plt.savefig(f"{paper_figure_path}/arpege_cost_grid.png", dpi=150, bbox_inches="tight")


# %%
def plot_costs_grid_with_flights(
    flights,
    colors,
    labels,
    df_cost,
    grid_color="Purples",
    ncols=3,
    nrows=2,
    legend_loc="lower right",
):
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        subplot_kw=dict(
            projection=ccrs.TransverseMercator(
                central_longitude=(LON0 + LON1) / 2,
                central_latitude=(LAT0 + LAT1) / 2,
            )
        ),
    )

    plot_time_step = flights[0].duration.total_seconds() / (ncols * nrows)

    for i, ax in enumerate(axes.flatten()):
        ax.set_extent([-20, 40, 25, 65])
        ax.add_feature(BORDERS, lw=0.5, edgecolor="gray")
        ax.coastlines(resolution="110m", lw=0.5, color="gray")

        current_flight_point = (
            flights[0]
            .data.query(
                f"{(i + 1) * plot_time_step}<=ts<={(i + 1) * plot_time_step + 600}"
            )
            .iloc[0]
        )

        grid_time_idx = np.argmin(
            abs(df_cost.timestamp.unique() - current_flight_point.timestamp)
        )
        sel_time = pd.to_datetime(df_cost.timestamp.unique()[grid_time_idx])

        grid_altitude_idx = np.argmin(
            abs(df_cost.height.unique() - current_flight_point.altitude * aero.ft)
        )
        sel_altitude = df_cost.height.unique()[grid_altitude_idx]

        df_cost_pivot = df_cost.query(
            f"height=={sel_altitude} and timestamp==@sel_time"
        ).pivot(index="latitude", columns="longitude", values="cost")

        lat, lon, val = (
            df_cost_pivot.index.values,
            df_cost_pivot.columns.values,
            df_cost_pivot.values,
        )

        ax.contourf(
            lon,
            lat,
            val,
            transform=proj,
            alpha=0.7,
            cmap=grid_color,
            vmin=0,
            vmax=df_cost.cost.quantile(0.99),
            antialiased=True,
            # edgecolors="face",
        )

        ax.text(
            0.03,
            0.97,
            f"Time: {current_flight_point.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"Grid Altitude: {int(round(current_flight_point.altitude, -2))} ft",
            transform=ax.transAxes,
            fontsize=14,
            va="top",
        )

        for r, p in flights[0].data.iloc[[0, -1]].iterrows():
            ax.scatter(p.longitude, p.latitude, c="k", transform=proj)

    for k, flight in enumerate(flights):
        for i, ax in enumerate(axes.flatten()):
            current_flight_path = flight.data.query(
                f"ts<={(i + 1) * plot_time_step + 600}"
            )
            remaining_flight_path = flight.data.query(
                f"ts>{(i + 1) * plot_time_step + 600}"
            )
            current_flight_point = current_flight_path.iloc[-1]

            grid_time_idx = np.argmin(
                abs(df_cost.timestamp.unique() - current_flight_point.timestamp)
            )
            sel_time = pd.to_datetime(df_cost.timestamp.unique()[grid_time_idx])

            grid_altitude_idx = np.argmin(
                abs(df_cost.height.unique() - current_flight_point.altitude * aero.ft)
            )
            sel_altitude = df_cost.height.unique()[grid_altitude_idx]

            df_cost_pivot = df_cost.query(
                f"height=={sel_altitude} and timestamp==@sel_time"
            ).pivot(index="latitude", columns="longitude", values="cost")

            # ax.scatter(
            #     current_flight_point.longitude,
            #     current_flight_point.latitude,
            #     color=colors[k],
            #     lw=2,
            #     transform=proj,
            # )

            ax.plot(
                current_flight_path.longitude,
                current_flight_path.latitude,
                color=colors[k],
                lw=2,
                transform=proj,
                label=labels[k],
            )

            ax.plot(
                remaining_flight_path.longitude,
                remaining_flight_path.latitude,
                color=colors[k],
                lw=1,
                ls="--",
                transform=proj,
            )

            ax.legend(loc=legend_loc, fontsize=11)

    plt.tight_layout()
    return plt


# %%
t_example = t_opt.query("icao24=='461fa0'")

# %%
# example

# from optimize import generate_optimal_flight

# t_example = (
#     Traffic(
#         generate_optimal_flight(
#             t_base["461fa0_08596_0"], debug=True, max_iteration=3000, max_nodes=60
#         ).data,
#     )
#     .resample("10s")
#     .eval()
# )

# t_example.to_parquet("data/example_flight.parquet")

# t_example = Traffic.from_file("data/example_flight.parquet")

# %%

t_example = Traffic(agg_contrail_conditions(arco_grid.interpolate(t_example.data)))


# %%
base_id = "461fa0_08596_0"

flight = t_example[base_id].assign(
    ts=lambda d: (d.timestamp - d.timestamp.iloc[0]).dt.total_seconds()
)

flight_opt_fuel = t_example[f"{base_id}_fuel"]

flight_opt_era5_03 = t_example[f"{base_id}_era5_03"]
flight_opt_era5_06 = t_example[f"{base_id}_era5_06"]
flight_opt_arpege_03 = t_example[f"{base_id}_arpege_03"]
flight_opt_arpege_06 = t_example[f"{base_id}_arpege_06"]

# plot_costs_grid_with_flights(
#     [flight], colors=["k"], labels=["flight"], df_cost=df_cost_era5
# )

# %%

colors = ["blue", "tab:blue", "tab:green", "k"]

labels = [
    "actual flight",
    "fuel optimal",
    "contrail (era5, c=0.3)",
    "contrail (era5, c=0.6)",
]

plot_costs_grid_with_flights(
    flights=[
        flight_opt_era5_03,
        flight_opt_era5_06,
        flight_opt_fuel,
        flight,
    ],
    colors=colors,
    labels=labels,
    df_cost=df_cost_era5,
    grid_color="Reds",
)

# plt.savefig(
#     f"{paper_figure_path}/optimization_example_1_era5.png",
#     dpi=150,
#     bbox_inches="tight",
# )

labels = [
    "actual flight",
    "fuel optimal",
    "contrail (arpege, c=0.3)",
    "contrail (arpege, c=0.6)",
]

plot_costs_grid_with_flights(
    flights=[
        flight_opt_arpege_03,
        flight_opt_arpege_06,
        flight_opt_fuel,
        flight,
    ],
    colors=colors,
    labels=labels,
    df_cost=df_cost_arpege,
    grid_color="Purples",
)

# plt.savefig(
#     f"{paper_figure_path}/optimization_example_1_arpege.png",
#     dpi=150,
#     bbox_inches="tight",
# )


# %%

flights = [
    flight,
    flight_opt_fuel,
    flight_opt_era5_03,
    flight_opt_era5_06,
]


colors = ["k", "tab:green", "tab:blue", "blue"]

grid_color = ["gray", "tab:green", "tab:orange", "tab:orange"]

labels = [
    "actual flight",
    "fuel optimal",
    "contrail optimal, era5, c=0.3",
    "contrail optimal, era5, c=0.6",
]


fig, axes = plt.subplots(4, 1, figsize=(6.5, 8), sharex=True, sharey=True)

for i, f in enumerate(flights):
    f = f.filter("aggressive").resample("10s")

    contrail_distance = int(
        sum([fc.distance() for fc in f.query("persistent").split("5min")])
    )

    ax = axes[i]
    ax.plot(f.data.timestamp, f.data.altitude, label=labels[i], color=colors[i])
    ax.scatter(
        f.data.query("persistent").timestamp,
        f.data.query("persistent").altitude,
        color=colors[i],
        s=30,
        label=f"actual contrail distance: {contrail_distance} nm",
    )

    ax.legend(loc="lower right", ncol=1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_label_coords(-0.1, 1.05)
    ax.grid(color=grid_color[i])

    if i == 0:
        ax.set_ylabel("altitude (ft)", rotation=0, ha="left")
        ax.set_ylim(0, 40_000)


ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

fig.autofmt_xdate()
plt.tight_layout()

# plt.savefig(
#     f"{paper_figure_path}/optimization_example_1_altitude_era5.png",
#     dpi=150,
#     bbox_inches="tight",
# )


# %%

base_id = "a0f427_18879_0"

flight_2 = t_opt[base_id].assign(
    ts=lambda d: (d.timestamp - d.timestamp.iloc[0]).dt.total_seconds()
)

flight_2_era5_03 = t_opt[f"{base_id}_era5_03"]
flight_2_era5_06 = t_opt[f"{base_id}_era5_06"]

flight_2_arpege_03 = t_opt[f"{base_id}_arpege_03"]
flight_2_arpege_06 = t_opt[f"{base_id}_arpege_06"]


# plot_costs_grid_with_flights(
#     [flight_2], colors=["k"], labels=["flight"], df_cost=df_cost_era5
# )


# flight.map_leaflet()

# %%
colors = [
    "tab:blue",
    "blue",
    "k",
]
labels = [
    # "fuel opt.",
    "contrail optimal (c=0.3)",
    "contrail optimal (c=0.6)",
    "actual flight",
]

plot_costs_grid_with_flights(
    flights=[
        # flight_opt_fuel,
        flight_2_era5_03,
        flight_2_era5_06,
        flight_2,
    ],
    colors=colors,
    labels=labels,
    df_cost=df_cost_era5,
    grid_color="Reds",
    ncols=3,
    nrows=1,
    legend_loc="lower left",
)


plot_costs_grid_with_flights(
    flights=[
        # flight_opt_fuel,
        flight_2_arpege_03,
        flight_2_arpege_06,
        flight_2,
    ],
    colors=colors,
    labels=labels,
    df_cost=df_cost_arpege,
    grid_color="RdPu",
    ncols=3,
    nrows=1,
    legend_loc="lower left",
)


# %%
colors = ["tab:blue", "blue"]
labels = [
    # "actual flight",
    # "fuel opt.",
    "contrail optimal (c=0.3)",
    "contrail optimal (c=0.6)",
]

plot_costs_grid_with_flights(
    flights=[
        # flight,
        # flight_opt_fuel,
        flight_2_era5_03,
        flight_2_era5_06,
    ],
    colors=colors,
    labels=labels,
    df_cost=df_cost_era5,
    grid_color="Reds",
    ncols=1,
    nrows=4,
    legend_loc="lower left",
)

# plt.savefig(
#     f"{paper_figure_path}/optimization_example_2_era5.png",
#     dpi=150,
#     bbox_inches="tight",
# )


plot_costs_grid_with_flights(
    flights=[
        # flight,
        # flight_opt_fuel,
        flight_2_arpege_03,
        flight_2_arpege_06,
    ],
    colors=colors,
    labels=labels,
    df_cost=df_cost_arpege,
    grid_color="RdPu",
    ncols=1,
    nrows=4,
    legend_loc="lower left",
)


# plt.savefig(
#     f"{paper_figure_path}/optimization_example_2_arpege.png",
#     dpi=150,
#     bbox_inches="tight",
# )

# %%

flights = [
    flight_2,
    flight_2_era5_03,
    flight_2_arpege_03,
    flight_2_era5_06,
    flight_2_arpege_06,
]

colors = ["k", "tab:blue", "blue", "tab:blue", "blue"]

grid_color = ["gray", "tab:orange", "tab:orange", "tab:purple", "tab:purple"]

labels = [
    "actual flight",
    "contrail optimal (era5, c=0.3)",
    "contrail optimal (arpege, c=0.3)",
    "contrail optimal (era5, c=0.6)",
    "contrail optimal (arpege, c=0.6)",
]


fig, axes = plt.subplots(5, 1, figsize=(7, 8), sharex=True, sharey=True)

for i, f in enumerate(flights):
    f = f.resample("10s")
    f = Flight(agg_contrail_conditions(arco_grid.interpolate(f.data)))

    contrail_distance = int(
        sum([fc.distance() for fc in f.query("persistent").split("5min")])
    )

    ax = axes[i]
    ax.plot(f.data.timestamp, f.data.altitude, label=labels[i], color=colors[i])
    ax.scatter(
        f.data.query("persistent").timestamp,
        f.data.query("persistent").altitude,
        color=colors[i],
        s=30,
        label=f"actual contrail distance: {contrail_distance} nm",
    )

    ax.legend(loc="lower right", ncol=1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_label_coords(-0.1, 1.05)
    ax.grid(color=grid_color[i], lw=2)

    if i == 0:
        ax.set_ylabel("altitude (ft)", rotation=0, ha="left")
        ax.set_ylim(15_000, 40_000)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

fig.autofmt_xdate()
plt.tight_layout()

# plt.savefig(
#     f"{paper_figure_path}/optimization_example_altitude_era5_vs_arpege.png",
#     dpi=150,
#     bbox_inches="tight",
# )


# %%

plt.figure(figsize=(7, 2))
plt.barh(
    [0, 1],
    [len((t_opt & t_base).flight_ids), len(t_base.flight_ids)],
    color=["tab:red", "tab:blue"],
    tick_label=["flights with contrails", "total flights"],
    height=0.4,
    zorder=100,
)
plt.ylim(-0.5, 1.5)
plt.grid()
plt.tight_layout()
plt.savefig(
    f"{paper_figure_path}/flights_vs_contrails.pdf", dpi=150, bbox_inches="tight"
)

# %%
print("Computing contrail distance consumption")


def compute_contrail_distance(flight: Flight):
    if flight.query("persistent") is None:
        return flight.assign(contrail_distance=0)

    fi = flight.query("persistent").split("10min")

    if fi.sum() == 0:
        return flight.assign(contrail_distance=0)

    contrail_distance = np.sum([f.distance() for f in fi])
    return flight.assign(contrail_distance=contrail_distance)


t_opt = t_opt.pipe(compute_contrail_distance).eval(max_workers=16, desc="processing")


# %%

all_flight_ids = t_opt.flight_ids

base_flight_ids = [
    i
    for i in all_flight_ids
    if ("fuel" not in i and "arpege" not in i and "era5" not in i)
]
fuel_flight_ids = [i for i in all_flight_ids if "fuel" in i]
f03_flight_ids = [i for i in all_flight_ids if "_03" in i]
f06_flight_ids = [i for i in all_flight_ids if "_06" in i]


stats_base = (
    t_opt.query(f"flight_id.isin({base_flight_ids})")
    .data.drop_duplicates("flight_id", keep="last")[
        ["flight_id", "contrail_distance", "total_fuel"]
    ]
    .set_index("flight_id")
)

stats_fuel = (
    t_opt.query(f"flight_id.isin({fuel_flight_ids})")
    .data.drop_duplicates("flight_id", keep="last")[
        ["flight_id", "contrail_distance", "total_fuel"]
    ]
    .set_index("flight_id")
)

stats_03 = (
    t_opt.query(f"flight_id.isin({f03_flight_ids})")
    .data.drop_duplicates("flight_id", keep="last")[
        ["flight_id", "contrail_distance", "total_fuel"]
    ]
    .set_index("flight_id")
)

stats_06 = (
    t_opt.query(f"flight_id.isin({f06_flight_ids})")
    .data.drop_duplicates("flight_id", keep="last")[
        ["flight_id", "contrail_distance", "total_fuel"]
    ]
    .set_index("flight_id")
)

# %%
results = []

for fid in tqdm(base_flight_ids):
    results.append(
        {
            "fid": fid,
            "base_contrail_distance": stats_base.loc[fid].contrail_distance,
            "fuel_opt_contrail_distance": stats_fuel.loc[
                f"{fid}_fuel"
            ].contrail_distance,
            "contrail_era5_03_contrail_distance": stats_03.loc[
                f"{fid}_era5_03"
            ].contrail_distance,
            "contrail_arpege_03_contrail_distance": stats_03.loc[
                f"{fid}_arpege_03"
            ].contrail_distance,
            "contrail_era5_06_contrail_distance": stats_06.loc[
                f"{fid}_era5_06"
            ].contrail_distance,
            "contrail_arpege_06_contrail_distance": stats_06.loc[
                f"{fid}_arpege_06"
            ].contrail_distance,
            "base_fuel": stats_base.loc[fid].total_fuel,
            "fuel_opt_fuel": stats_fuel.loc[f"{fid}_fuel"].total_fuel,
            "contrail_era5_03_fuel": stats_03.loc[f"{fid}_era5_03"].total_fuel,
            "contrail_arpege_03_fuel": stats_03.loc[f"{fid}_arpege_03"].total_fuel,
            "contrail_era5_06_fuel": stats_06.loc[f"{fid}_era5_06"].total_fuel,
            "contrail_arpege_06_fuel": stats_06.loc[f"{fid}_arpege_06"].total_fuel,
        }
    )

# %%
df = pd.DataFrame.from_dict(results)

# df.to_csv("data/stats_results.csv", index=False)

df

# %%
df = pd.read_csv("data/stats_results.csv")

# %%
plt.figure(figsize=(6.5, 3))

plt.boxplot(
    [
        df.base_contrail_distance,
        df.fuel_opt_contrail_distance,
        df.contrail_era5_03_contrail_distance,
        df.contrail_era5_06_contrail_distance,
    ],
    labels=[
        "original flight",
        "fuel opt.",
        "contrail opt, era5, c=0.3",
        "contrail opt, era5, c=0.6",
    ],
    vert=False,
    notch=True,
)
plt.gca().tick_params(axis="y", labelsize=16)
plt.axhline(2.5, color="gray", lw=1, ls="--")

plt.grid(axis="x")
plt.xticks(np.arange(0, 1200, 150))

plt.xlabel("contrail distance (nm)")

plt.gca().invert_yaxis()

# plt.savefig(
#     f"{paper_figure_path}/contrail_distance_era5_boxplot.png",
#     dpi=150,
#     bbox_inches="tight",
# )


# %%

plt.figure(figsize=(6.5, 4))

plt.boxplot(
    [
        df.base_contrail_distance,
        df.fuel_opt_contrail_distance,
        df.contrail_era5_03_contrail_distance,
        df.contrail_arpege_03_contrail_distance,
        df.contrail_era5_06_contrail_distance,
        df.contrail_arpege_06_contrail_distance,
    ],
    labels=[
        "original flight",
        "fuel opt.",
        "contrail opt, era5, c=0.3",
        "contrail opt, arpege, c=0.3",
        "contrail opt, era5, c=0.6",
        "contrail opt, arpege, c=0.6",
    ],
    vert=False,
    notch=True,
)

plt.axhline(2.5, color="gray", lw=1, ls="--")
plt.axhline(4.5, color="gray", lw=1, ls="--")

plt.grid(axis="x")
plt.xticks(np.arange(0, 1200, 150))
plt.gca().tick_params(axis="y", labelsize=16)

plt.xlabel("contrail distance (nm)")

plt.gca().invert_yaxis()

# plt.savefig(
#     f"{paper_figure_path}/contrail_distance_boxplot.png",
#     dpi=150,
#     bbox_inches="tight",
# )


# %%
import seaborn as sns

plt.figure(figsize=(5, 4))

df_plot = (
    pd.DataFrame()
    .assign(
        d_fuel=(df.contrail_era5_03_fuel - df.fuel_opt_fuel) / df.fuel_opt_fuel,
        d_dist=df.contrail_era5_03_contrail_distance - df.fuel_opt_contrail_distance,
    )
    .query("-0.05<d_fuel<0.05")
    .query("-100<d_dist<100")
)

sns.jointplot(
    data=df_plot, x="d_dist", y="d_fuel", kind="hist", color="tab:blue", bins=81
)
plt.axvline(0, color="gray", lw=1, ls="--")
plt.axhline(0, color="gray", lw=1, ls="--")
plt.ylim(-0.1, 0.1)
plt.xlim(-100, 100)
plt.xlabel("contrail distance difference per flight (nm)")
plt.ylabel("fuel consumption difference per flight (%)")

plt.tight_layout()

plt.savefig(
    f"{paper_figure_path}/contrail_distance_vs_fuel_03.png",
    dpi=150,
    bbox_inches="tight",
)

# %%

plt.figure(figsize=(5, 4))

df_plot = (
    pd.DataFrame()
    .assign(
        d_fuel=(df.contrail_era5_06_fuel - df.fuel_opt_fuel) / df.fuel_opt_fuel,
        d_dist=df.contrail_era5_06_contrail_distance - df.fuel_opt_contrail_distance,
    )
    .query("-0.1<d_fuel<0.1")
    .query("-100<d_dist<100")
)

sns.jointplot(
    data=df_plot, x="d_dist", y="d_fuel", kind="hist", color="midnightblue", bins=80
)
plt.axvline(0, color="gray", lw=1, ls="--")
plt.axhline(0, color="gray", lw=1, ls="--")
plt.ylim(-0.1, 0.1)
plt.xlim(-100, 100)
plt.xlabel("contrail distance difference per flight (nm)")
plt.ylabel("fuel consumption difference per flight (%)")

plt.tight_layout()
plt.savefig(
    f"{paper_figure_path}/contrail_distance_vs_fuel_06.png",
    dpi=150,
    bbox_inches="tight",
)


# %%

lat0, lon0 = 40.8, -8
lat1, lon1 = 52.7, 16.2


# %%

all_flight_ids = t_opt.flight_ids
base_flight_ids = [i for i in all_flight_ids if "opt" not in i]

t_opt_midday = (
    t_opt.query(
        f"timestamp.dt.hour==10 and timestamp.dt.minute<45 and flight_id.isin({base_flight_ids})"
    )
    .query(f"{lat0}<latitude<{lat1} and {lon0}<longitude<{lon1}")
    .longer_than("5min")
    .resample("10s")
    .eval()
)

t_opt_midday = Traffic(
    agg_contrail_conditions(arco_grid.interpolate(t_opt_midday.data))
)

# %%


def split_flights(flight: Flight):
    return flight.split("5min").all("{self.flight_id}_{i}")


t_persistent = (
    t_opt_midday.query("persistent").pipe(split_flights).resample("1 min").eval()
)
t_persistent


# t_persistent.map_leaflet()

from cartes.crs import Lambert93

proj = Lambert93()

fig, ax = plt.subplots(
    1,
    1,
    figsize=(10, 6),
    subplot_kw=dict(
        projection=ccrs.TransverseMercator(
            central_longitude=(lon0 + lon1) / 2,
            central_latitude=(lat0 + lat1) / 2,
        )
    ),
)

ax.set_extent([lon0, lon1, lat0, lat1])
ax.add_feature(BORDERS, lw=0.5, color="gray")
ax.coastlines(resolution="50m", lw=0.5, color="gray")

heights = df_cost_era5.height.unique()

df_cost_pivot = df_cost_era5.query(
    f"height=={heights[9]} and timestamp.dt.hour==11"
).pivot(index="latitude", columns="longitude", values="cost")

lat, lon, val = (
    df_cost_pivot.index.values,
    df_cost_pivot.columns.values,
    df_cost_pivot.values,
)

ax.contourf(
    lon,
    lat,
    val,
    transform=ccrs.PlateCarree(),
    alpha=0.3,
    cmap="Reds",
    vmin=-0.2,
    vmax=df_cost_era5.cost.quantile(0.99),
    antialiased=True,
    # edgecolors="face",
)

for k, flight in enumerate(t_persistent):
    ax.plot(
        flight.data.longitude,
        flight.data.latitude,
        color="tab:red",
        lw=1,
        transform=ccrs.PlateCarree(),
    )

# remove all geo spines
ax.spines["geo"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{paper_figure_path}/sat_image_flights.png", dpi=150, bbox_inches="tight")


# %%
