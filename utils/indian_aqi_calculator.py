import math

# CPCB Breakpoints (24hr avg)
BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500),
    ],
    "co": [
        (0, 1, 0, 50),
        (1.1, 2, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 50, 401, 500),
    ],
}


# Outlier caps
OUTLIER_LIMITS = {
    "pm25": 1000,
    "pm10": 2000,
    "no2": 500,
    "o3": 500,
    "so2": 2000,
    "co": 50,
}


def calculate_sub_index(pollutant, concentration):

    if concentration is None:
        return None

    if concentration < 0:
        return None

    if pollutant in OUTLIER_LIMITS and concentration > OUTLIER_LIMITS[pollutant]:
        return None

    for bp_lo, bp_hi, i_lo, i_hi in BREAKPOINTS[pollutant]:
        if bp_lo <= concentration <= bp_hi:
            return round(
                ((i_hi - i_lo) / (bp_hi - bp_lo)) *
                (concentration - bp_lo) + i_lo
            )

    return None


def calculate_indian_aqi(data_dict):

    sub_indices = {}

    for pollutant in BREAKPOINTS.keys():
        value = data_dict.get(pollutant)
        sub_index = calculate_sub_index(pollutant, value)

        if sub_index:
            sub_indices[pollutant] = sub_index

    if len(sub_indices) < 3:
        return None, None, None

    if "pm25" not in sub_indices and "pm10" not in sub_indices:
        return None, None, None

    final_aqi = max(sub_indices.values())
    dominant_pollutant = max(sub_indices, key=sub_indices.get)

    return final_aqi, dominant_pollutant, sub_indices