# Air Quality & Mobility Nowcasting – Project Overview

## Problem statement
We want to nowcast current air quality and mobility trends for selected cities and test “what-if” scenarios (e.g., traffic reduction, weather changes) to see short-term impacts on air quality.

## Objectives
- Ingest air quality and mobility data for chosen cities.
- Clean, align, and join datasets on time and location.
- Build a baseline nowcasting model.
- Provide simple “what-if” knobs (e.g., percent mobility change) and show predicted AQ effects.
- Output clear visuals and a short report.

## Scope (initial)
- Cities: start with 2–3 cities to keep it manageable.
- Time granularity: hourly or daily, depending on data reliability.
- Metrics: PM2.5, PM10, NO2 (adjust as data allows), plus mobility indexes.

## Data sources (planned)
- Air quality: OpenAQ (API/export)
- Mobility: your chosen public source (to finalize in the next step)
- Weather (optional but useful): temperature, wind, humidity

## Folder map
- `scripts/` Python modules and runnable scripts
- `data/` raw and interim datasets (ignored by git)
- `results/` figures, models, reports (ignored by git)
- `docs/` documentation and notes

## Next steps
1) Decide cities and date range
2) Write `scripts/fetch_data.py` to pull raw data
3) Write `scripts/preprocess.py` to clean/merge datasets
4) Write `scripts/train_model.py` for a first baseline
5) Write `scripts/what_if.py` for scenario experiments
6) Save outputs to `results/`
