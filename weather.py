import streamlit as st
import requests

# Set up the app title
st.title("Weather Forecast App")

# Define the API endpoint and API key
url = "https://api.openweathermap.org/data/2.5/weather"
api_key = "0577598accdd80c30eedeb2967e88368"
# Create a function to get the weather data for a given city
def get_weather(city):
    params = {"appid": api_key, "q": city, "units": "metric"}
    response = requests.get(url, params=params)
    data = response.json()
    return data

# Create the user interface
city = st.text_input("Enter a city name")
if city:
    data = get_weather(city)
    if data["cod"] == 200:
        st.write(f"Weather for {city.capitalize()}:")
        st.write(f"Temperature: {data['main']['temp']} °C")
        st.write(f"Feels like: {data['main']['feels_like']} °C")
        st.write(f"Humidity: {data['main']['humidity']} %")
        st.write(f"Description: {data['weather'][0]['description'].capitalize()}")
    else:
        st.write(f"Error: {data['message']}")
