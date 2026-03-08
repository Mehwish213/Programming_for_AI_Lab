from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_KEY = "e28df515aa044ae59f7154956250309"

@app.route("/", methods=["GET", "POST"])
def index():
    weather_data = None
    error_message = None

    if request.method == "POST":
        city = request.form.get("cityInput", "").strip()
        if city:
            url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"
            try:
                response = requests.get(url)
                data = response.json()
                if "error" in data:
                    error_message = data["error"]["message"]
                else:
                    weather_data = {
                        "location": f"{data['location']['name']}, {data['location']['country']}",
                        "temp_c": data["current"]["temp_c"],
                        "condition": data["current"]["condition"]["text"],
                        "icon": "https:" + data["current"]["condition"]["icon"]
                    }
            except Exception as e:
                error_message = "Unable to fetch weather data. Try again later."
        else:
            error_message = "Please enter a city name."

    return render_template("index.html", weather=weather_data, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)