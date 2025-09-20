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

##Integrated Parameters for AQI + Mobility Prediction & Scenario Modeling
1. Temporal & Seasonal Factors

Day of week (weekday vs weekend)

Public holidays & festivals (mobility pattern change)

Season (winter, summer, monsoon → AQI drastically varies)

2. Weather / Meteorological Conditions

Wind speed & direction 🌬

Temperature 🌡 (incl. inversion events)

Humidity 💧

Rainfall ☔

Visibility (fog/smog conditions)

3. Mobility & Transport Parameters

Traffic congestion index 🚦

Vehicle count per hour/day

Vehicle type share (cars, buses, trucks, 2-wheelers, EVs)

Average speed of traffic flow

Public transport usage % vs private vehicle usage %

Fuel type distribution (petrol, diesel, CNG, EV)

4. Emission & Pollution Sources

Industrial activity index (NO₂, SO₂, VOCs, CO₂)

Construction activity intensity 🚧 (dust, PM emissions)

Burning practices 🔥 (crop residue, garbage, firecrackers)

Energy consumption mix (fossil fuels vs renewables)

5. Urban & Geographical Conditions

Population density 👥

Road density / traffic hotspots

Land use distribution (residential, industrial, commercial)

Green cover % 🌳 (parks, urban trees)

Proximity to highways / industrial zones

Topography 🏔 (valleys, flat, coastal areas → pollutant trapping vs dispersion)

6. Policy & Intervention Inputs

Odd-even vehicle rule days

Construction bans (govt-imposed during high AQI days)

Industrial shutdown days

Public campaigns (e.g., car-free days, EV push)

Fuel price shocks (indirect effect on mobility)

7. Health & Socio-economic Impact (Optional, but adds depth)

Respiratory illness/hospital visits (linked to AQI)

Mortality/morbidity estimates from pollution

Productivity loss (school/work absences during bad AQI)

Economic cost of pollution (optional, secondary analysis)


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
