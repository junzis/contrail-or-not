# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from traffic.core import Flight, Traffic


# %%
def split_flight_legs(flight: Flight) -> None | Flight:
    # recreate flight ids with different legs
    above_fl150 = flight.query("altitude > 25000")
    if above_fl150 is not None:
        return above_fl150.split("30 min").all("{self.flight_id}_{i}")
    return None


def filter_trajectory(flight):
    flight = flight.query("altitude<45_000 and -150<vertical_rate<150")

    if flight is None:
        return None

    cruise = flight.phases().query("phase=='CRUISE'")
    if cruise is None:
        return None

    start = cruise.start
    stop = cruise.stop
    flight = flight.between(start, stop)

    if flight is None:
        return None

    flight = flight.filter("kalman")

    return flight


# %%
t_raw = Traffic.from_file("data/eu_flights_2022feb20_raw.parquet.gz")


# %%
print("processing flights")

t = t_raw.drop(["last_position", "onground"], axis=1).query("latitude.notnull()")

t = (
    t.iterate_lazy(iterate_kw=dict(by="45min"))
    .assign_id(name="{self.icao24}_{idx:>05}")
    .pipe(split_flight_legs)
    .longer_than("30 minutes")
    .shorter_than("8 hours")
    .eval(max_workers=16, desc="processing")
)

t = t.pipe(filter_trajectory).eval(max_workers=16, desc="filtering")

t = t.resample("30s").eval(max_workers=16, desc="resampling")


# %%
from fastmeteo.source import ArcoEra5
from openap import aero, contrail

aircraft = pd.read_csv("data/aircraft_db.csv")

t = t.merge(aircraft, on="icao24", how="left")

print("running fastmeteo")

ARCO_GRID = ArcoEra5(local_store="data/era5-zarr/", model_levels=137)


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


t = Traffic(agg_contrail_conditions(ARCO_GRID.interpolate(t.data)))

# %%
t.to_parquet("data/eu_flights_2022feb20_filter_resample_meteo.parquet.gz", index=False)
