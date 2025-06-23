# %%
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastmeteo.source import ArcoEra5
from openap import FuelFlow, aero, contrail, top
from traffic.core import Flight, Traffic

# %%

ALT_BOUND = 2000
TYPECODE = "A320"


# %%
def calc_fuel(flight):
    if isinstance(flight, pd.DataFrame):
        flight = Flight(flight)

    fuelflow = FuelFlow(TYPECODE)
    mass0 = flight.mass0_max

    if "tas" in flight.data.columns and flight.tas_max > 0:
        tas = flight.data.tas
    else:
        vg = flight.data.groundspeed * aero.kts
        psi = np.radians(flight.data.track)
        vgx = vg * np.sin(psi)
        vgy = vg * np.cos(psi)
        vax = vgx - flight.data.u_component_of_wind
        vay = vgy - flight.data.v_component_of_wind
        tas = np.sqrt(vax**2 + vay**2) / aero.kts

        tas = np.where(tas == tas, tas, flight.data.groundspeed)
        flight = flight.assign(tas=tas)

    mass = mass0 * np.ones_like(tas)
    dt = flight.data.timestamp.diff().bfill().dt.total_seconds()

    # fast way to calculate fuel flow without iterate over rows
    for i in range(6):
        ff = fuelflow.enroute(
            mass=mass,
            tas=tas,
            alt=flight.data.altitude,
            vs=flight.data.vertical_rate,
        )
        fuel = ff * dt
        mass[1:] = mass0 - fuel.cumsum()[:-1]

    flight = flight.assign(
        fuel_flow=ff,
        fuel=fuel,
        total_fuel=fuel.sum(),
        mass=mass,
    )

    return flight


def compute_contrail_time(flight: Flight):
    if flight.query("persistent") is None:
        return 0

    fi = flight.query("persistent").split("10min")

    if fi.sum() == 0:
        return 0

    contrail_time = np.sum([f.duration for f in fi]).total_seconds() // 60
    return contrail_time


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

t = Traffic.from_file("data/eu_flights_2022feb20_filter_resample_meteo.parquet.gz")

file_era5_cost = "data/grid_era5_smoothed.parquet.gz"
df_cost_era5 = pd.read_parquet(file_era5_cost)


file_arpege_cost = "data/grid_arpege_smoothed.parquet.gz"
df_cost_arpege = pd.read_parquet(file_arpege_cost).fillna(0)

wind_era5 = df_cost_era5[
    ["ts", "timestamp", "latitude", "longitude", "altitude", "height", "u", "v"]
].eval("h=height")

wind_arpege = df_cost_arpege[
    ["ts", "timestamp", "latitude", "longitude", "altitude", "height", "u", "v"]
].eval("h=height")

# %%
# flight = t["461fa0_08596_0"]


# %%
def generate_optimal_flight(flight, debug=False, max_iteration=3000, max_nodes=60):
    # %%
    def df_to_flight(df, appendix):
        return calc_fuel(
            Flight(
                df.assign(
                    timestamp=lambda x: flight.start + pd.to_timedelta(x.ts, unit="s"),
                    icao24=flight.icao24,
                    callsign=flight.callsign,
                    typecode=flight.typecode,
                    registration=flight.registration,
                    operator=flight.operator_max,
                    flight_id=flight.flight_id + appendix,
                    mass0=mass0,
                )
            )
        )

    def estimate_mass(flight):
        dist_km = flight.distance() * aero.nm / 1000
        crusie_alt = flight.altitude_median
        mass0 = 1.8533 * dist_km - 1.99133 * crusie_alt + 133497
        mass0 = mass0 - 2000  # remove the climbing fuel
        mass0 = min(mass0, 76000)
        return mass0

    # %%
    contrail_time = compute_contrail_time(flight)

    # %%
    if contrail_time < 5:
        return None

    # %%

    mass0 = estimate_mass(flight)
    m0 = mass0 / 78000

    flight = calc_fuel(flight.assign(mass0=mass0))

    # %%
    flight_alt_min_bound = flight.data.altitude.quantile(0.10) - ALT_BOUND
    flight_alt_max_bound = flight.data.altitude.quantile(0.90) + ALT_BOUND

    # %%
    start_day = flight.start.floor("1d")
    start_seconds = (flight.start - start_day).total_seconds()

    grid_start_time = flight.start - pd.Timedelta("1h")
    grid_stop_time = flight.stop + pd.Timedelta("1h")

    wind_era5_flight = wind_era5.query(
        f"'{grid_start_time}'<=timestamp<='{grid_stop_time}'"
    ).assign(ts=lambda x: (x.ts - start_seconds))

    wind_arpege_flight = wind_arpege.query(
        f"'{grid_start_time}'<=timestamp<='{grid_stop_time}'"
    ).assign(ts=lambda x: (x.ts - start_seconds))

    interpolant_era5 = top.tools.interpolant_from_dataframe(
        df_cost_era5.query(
            f"'{grid_start_time}'<=timestamp<='{grid_stop_time}'"
        ).assign(ts=lambda x: (x.ts - start_seconds))
    )

    interpolant_arpege = top.tools.interpolant_from_dataframe(
        df_cost_arpege.query(
            f"'{grid_start_time}'<=timestamp<='{grid_stop_time}'"
        ).assign(ts=lambda x: (x.ts - start_seconds))
    )

    # %%
    optimizer = top.Cruise(
        actype=TYPECODE,
        origin=flight.data[["latitude", "longitude"]].iloc[0].tolist(),
        destination=flight.data[["latitude", "longitude"]].iloc[-1].tolist(),
        m0=m0,
    )

    optimizer.setup(debug=debug, max_iteration=max_iteration, max_nodes=max_nodes)
    optimizer.enable_wind(wind_era5_flight)

    # optimizer.fix_cruise_altitude()

    flight_top_fuel = optimizer.trajectory(
        objective="fuel",
        h_min=flight_alt_min_bound * aero.ft,
        h_max=flight_alt_max_bound * aero.ft,
    )
    flight_top_fuel = flight_top_fuel.assign(
        solver_stats=optimizer.solver.stats()["unified_return_status"],
        solver_iterations=optimizer.solver.stats()["iter_count"],
    )

    flight_opt_fuel = df_to_flight(flight_top_fuel, "_fuel")

    # %%
    def objective(x, u, dt, coef, **kwargs):
        grid_cost = optimizer.obj_grid_cost(
            x, u, dt, time_dependent=True, n_dim=4, **kwargs
        )
        fuel_cost = optimizer.obj_fuel(x, u, dt, **kwargs)
        return grid_cost * coef + fuel_cost * (1 - coef)

    def optimize_one(grid_name, coef):
        if grid_name == "era5":
            wind_grid = wind_era5_flight
            interpolant = interpolant_era5
        elif grid_name == "arpege":
            wind_grid = wind_arpege_flight
            interpolant = interpolant_arpege

        optimizer.enable_wind(wind_grid)
        df_optimized = optimizer.trajectory(
            objective=objective,
            interpolant=interpolant,
            initial_guess=flight_top_fuel,
            h_min=flight_alt_min_bound * aero.ft,
            h_max=flight_alt_max_bound * aero.ft,
            coef=coef,
        )
        df_optimized = df_optimized.assign(
            solver_stats=optimizer.solver.stats()["unified_return_status"],
            solver_iterations=optimizer.solver.stats()["iter_count"],
        )
        flight_optimized = df_to_flight(df_optimized, f"_{grid_name}_0{int(coef * 10)}")
        return flight_optimized

    flight_opt_era5_03 = optimize_one("era5", 0.3)
    flight_opt_era5_06 = optimize_one("era5", 0.6)
    flight_opt_arpege_03 = optimize_one("arpege", 0.3)
    flight_opt_arpege_06 = optimize_one("arpege", 0.6)

    # %%
    # contrail_time_optimal = compute_contrail_time(
    #     Flight(agg_contrail_conditions((fmg.interpolate(flight_contrail_optimal.data))))
    # )
    # print(f"{flight.flight_id}: optimal contrail time: {contrail_time_optimal} minutes")

    # fuel_margin = flight_contrail_optimal.fuel_sum / flight_opt_fuel.fuel_sum
    # print(f"{flight.flight_id}: contrail optimal fuel factor: {fuel_margin}")

    # flight_contrail_optimal.map_leaflet()

    # plot_costs_grid_with_flight(flight_contrail_optimal, df_cost)

    gc.collect()

    return Flight(
        pd.concat(
            [
                flight.data,
                flight_opt_fuel.data,
                flight_opt_era5_03.data,
                flight_opt_era5_06.data,
                flight_opt_arpege_03.data,
                flight_opt_arpege_06.data,
            ]
        )
    )


def failsafe_generate_optimal_flight(flight):
    try:
        return generate_optimal_flight(flight)
    except Exception as e:
        print(e)
        return None


# %%
# flight_res = generate_optimal_flight(t["461fa0_08596_0"])


# %%

# for flight in t:
#     print(flight.flight_id)
#     generate_optimal_flight(flight)


# %%
if __name__ == "__main__":
    print("Processing flights")

    t_opt = (
        t.iterate_lazy(tqdm_kw={"ncols": 0})
        .pipe(failsafe_generate_optimal_flight)
        .eval(
            max_workers=16,
            desc="optimizing flights",
            cache_file="data/all_optimized.parquet",
        )
    )

    t_opt = t_opt.resample("10s").eval(max_workers=16, desc="resampling")

    # %%
    print("running fastmeteo")
    arco_grid = ArcoEra5(local_store="data/era5-zarr/", model_levels=137)

    t_opt = Traffic(agg_contrail_conditions(arco_grid.interpolate(t_opt.data)))

    t_opt.to_parquet("data/all_optimized_resampled.parquet.gz", index=False)
