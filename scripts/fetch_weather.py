import requests

API_KEY = "ab1c6a5cd00c250e7f9621ba1ef2ed67"
city = "Indore"

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

try:
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        print(f"City: {data['name']}")
        print(f"Temperature: {data['main']['temp']}°C")
        print(f"Weather: {data['weather'][0]['description']}")
        print(f"Humidity: {data['main']['humidity']}%")
        print(f"Wind Speed: {data['wind']['speed']} m/s")
    else:
        print(f"Error {data.get('cod')}: {data.get('message')}")
except Exception as e:
    print(f"Exception occurred: {e}")
