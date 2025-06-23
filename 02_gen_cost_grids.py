# %%
from datetime import timedelta
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from fastmeteo.source import ArcoEra5
from openap import aero, contrail
from scipy.ndimage import gaussian_filter

# %%
LAT0, LAT1 = 26, 66
LON0, LON1 = -15, 40
DATE = pd.to_datetime("2022-02-20", utc=True)

ARCO_GRID = ArcoEra5(local_store="data/era5-zarr/", model_levels=137)


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


def smooth_cost_grid(df_cost):
    print("Smoothing grid")
    cost = df_cost.cost.values.reshape(
        df_cost.ts.nunique(),
        df_cost.height.nunique(),
        df_cost.latitude.nunique(),
        df_cost.longitude.nunique(),
    )

    cost_ = gaussian_filter(cost, sigma=(0, 2, 2, 2), mode="nearest")
    df_cost = df_cost.assign(cost=cost_.flatten()).fillna(0)
    return df_cost


# %%
print("Generating cost grid coordinates")

lats = np.arange(LAT0, LAT1, 0.5)
lons = np.arange(LON0, LON1, 0.5)
alts = np.arange(20_000, 45_000, 2000)
timestamps = pd.date_range(DATE, DATE + pd.Timedelta("1d"), freq="1h")

Timestamps, Alts, Lats, Lons = np.meshgrid(timestamps, alts, lats, lons)

GRID = pd.DataFrame(
    {
        "timestamp": Timestamps.ravel(),
        "altitude": Alts.ravel(),
        "latitude": Lats.ravel(),
        "longitude": Lons.ravel(),
    }
)


# %%
print("Generating ERA5 cost grid")


def generate_era5_cost():
    era5_grid = ARCO_GRID.interpolate(GRID)

    era5_grid = agg_contrail_conditions(era5_grid)

    df_cost = (
        era5_grid.assign(
            height=lambda x: x.altitude * aero.ft,
            cost=lambda x: x.persistent.astype(float),
            ts=lambda x: (x.timestamp - x.timestamp.iloc[0]).dt.total_seconds(),
        )
        .rename(columns={"u_component_of_wind": "u", "v_component_of_wind": "v"})
        .sort_values(["ts", "height", "latitude", "longitude"])
    )

    return df_cost


file_era5_cost = "data/grid_era5_smoothed.parquet.gz"

df_cost_era5 = generate_era5_cost()
df_cost_era5 = smooth_cost_grid(df_cost_era5)
df_cost_era5.to_parquet(file_era5_cost, index=False)

# %%
print("Generating ARPEGE cost grid")

file = "data/arpege/ARPEGE_0.1_IP1_00H12H_20220220T0000.grib2"


def generate_arpege_cost():
    grib_files = sorted(glob("data/arpege/ARPEGE*.grib2"))

    dss = []
    for gf in grib_files:
        ds_grid = xr.open_mfdataset(gf, engine="cfgrib").sel(
            longitude=slice(LON0, LON1, 2),
            latitude=slice(LAT1, LAT0, 2),
        )

        for h in range(6):
            dss.append(ds_grid.sel(step=timedelta(hours=h)).drop("step"))

    ds_arpege = xr.concat(dss, dim="valid_time").drop("time")

    coords = {
        "valid_time": (("points",), GRID.timestamp.to_numpy(dtype="datetime64[ns]")),
        "latitude": (("points",), GRID.latitude.values),
        "longitude": (("points",), GRID.longitude.values),
        "isobaricInhPa": (
            ("points",),
            aero.pressure(GRID.altitude * aero.ft) / 100,
        ),
    }

    ds_grid = xr.Dataset(coords=coords)

    meteo_params = (
        ds_arpege.interp(
            ds_grid.coords,
            method="linear",
            assume_sorted=False,
            kwargs={"fill_value": None},
        )
        .to_dataframe()
        .reset_index()[["t", "r", "u", "v"]]
    )

    df_cost = pd.concat([GRID, meteo_params], axis=1).assign(
        rhi=lambda d: contrail.rhw2rhi(d.r / 100, d.t),
        crit_temp=lambda d: contrail.critical_temperature_water(
            aero.pressure(d.altitude * aero.ft)
        ),
        sac=lambda d: d.t < d.crit_temp,
        issr=lambda d: d.rhi > 1,
        persistent=lambda d: d.sac & d.issr,
    )

    df_cost = df_cost.assign(
        height=lambda x: x.altitude * aero.ft,
        cost=lambda x: x.persistent.astype(float),
        ts=lambda x: (x.timestamp - x.timestamp.iloc[0]).dt.total_seconds(),
    ).sort_values(["ts", "height", "latitude", "longitude"])

    return df_cost


file_arpege_cost = "data/grid_arpege_smoothed.parquet.gz"

df_cost = generate_arpege_cost()
df_cost_arpege = smooth_cost_grid(df_cost)
df_cost_arpege.to_parquet(file_arpege_cost, index=False)

# %%
