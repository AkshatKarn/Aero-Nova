def get_category_with_color(aqi: int):
    """
    Returns AQI category name and associated hex color.
    Based on standard AQI scale.
    """

    if aqi is None:
        return "Unknown", "#9E9E9E"

    try:
        aqi = int(aqi)
    except (ValueError, TypeError):
        return "Invalid", "#9E9E9E"

    if 0 <= aqi <= 50:
        return "Good 🟢", "#00E400"
    elif 51 <= aqi <= 100:
        return "Moderate 🟡", "#FFFF00"
    elif 101 <= aqi <= 150:
        return "Unhealthy for Sensitive Groups 🟠", "#FF7E00"
    elif 151 <= aqi <= 200:
        return "Unhealthy 🔴", "#FF0000"
    elif 201 <= aqi <= 300:
        return "Very Unhealthy 🟣", "#8F3F97"
    elif aqi > 300:
        return "Hazardous ⚫", "#7E0023"
    else:
        return "Out of Range", "#9E9E9E"
