# %%
from glob import glob

import pandas as pd
from fastmeteo.source import ArcoEra5
from openap import aero, contrail
from traffic.core import Traffic

pd.options.display.max_columns = 100


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


t_new = Traffic(pd.concat([pd.read_parquet(f) for f in glob("data/optimized/*.gz")]))

t_new = t_new.resample("10s").eval(max_workers=16, desc="resampling")

# %%

print("running fastmeteo")
arco_grid = ArcoEra5(local_store="data/era5-zarr/", model_levels=137)

t_new = Traffic(agg_contrail_conditions(arco_grid.interpolate(t_new.data)))

# %%
t_new.to_parquet("data/all_optimized_resampled.parquet.gz", index=False)

# %%
