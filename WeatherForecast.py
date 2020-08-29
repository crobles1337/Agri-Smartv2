#OpenWeather One Call API 

import requests
import datetime



Key = '35e02b4dab539973840fc771425f3539'
lat = '48.3670'
lon = '18.0764'
exclude = 'minutely' # current, minutely, hourly, daily
args = [lat, lon, exclude, Key]
tempunit = 'metric' #imperial, auto is Kelvin

Days = ['Sunday', 'Monday,', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def main():
    WeatherCall(lat, lon)

def WeatherCall(lat, lon, daily = True, hourly=False, minutely = False, current=False, tempunit = 'metric'):
    key = '35e02b4dab539973840fc771425f3539'
    if daily == True:
        WeatherStats = WeekForecast(lat, lon, key, tempunit)
        
        dayofweek = datetime.datetime.now().weekday()
        for i in range(7):
            print('The forecast for ', Days[dayofweek+i], 'is' )
            for k in WeatherStats:
                print(k, ': ', WeatherStats[k][i], WeatherStats[k][7]) 


def WeekForecast(lat, lon, key, tempunit):
    exclude = 'minutely, current, hourly'
    WeatherData = requests.get('https://api.openweathermap.org/data/2.5/onecall?lat={}&lon={}&units={}&exclude={}&appid={}'.format(lat, lon, tempunit,exclude, Key))
    Weather = WeatherData.json()
    DailyW = Weather['daily']
    if tempunit == 'metric':
        TempUnit = 'celsius'
    Rain = [0]*8
    Rain[7] = 'mm'
    MinTemp = [0]*8
    MinTemp[7] = TempUnit
    MaxTemp = [0]*8
    MaxTemp[7] = TempUnit
    Humid = [0]*8
    Humid[7] = '%'
    DewPoint = [0]*8
    DewPoint[7] = TempUnit
    Clouds = [0]*8
    Clouds[7] = '%'
    Descrip = [0]*8
    Descrip[7] = ''
    UVIs = [0]*8
    UVIs[7] = ''
    Morn = [0]*8
    Morn[7] = TempUnit
    Night = [0]*8
    Night[7] = TempUnit
    Day = [0]*8
    Day[7] = TempUnit
    WindSpeed = [0]*8
    WindSpeed[7] = 'meter/sec'
    WindDeg = [0]*8
    WindDeg[7] = 'degrees'
    Eve = [0]*8
    Eve[7] = TempUnit

    for i in range(7):
        Day[i] = str(DailyW[i]['temp']['day'])
        Morn[i] = DailyW[i]['temp']['morn']
        Eve[i] = DailyW[i]['temp']['eve']
        Night[i] = DailyW[i]['temp']['night']
        MaxTemp[i] = DailyW[i]['temp']['max']
        MinTemp[i] = DailyW[i]['temp']['min']
        Humid[i] = DailyW[i]['humidity']
        DewPoint[i] = DailyW[i]['dew_point']
        WindSpeed[i] = DailyW[i]['wind_speed']
        WindDeg[i] = DailyW[i]['wind_deg']
        Descrip[i] = DailyW[i]['weather'][0]['description']
        UVIs[i] = DailyW[i]['uvi']
        Clouds[i] = DailyW[i]['clouds']

        if 'rain' in DailyW[i]:
            print("Rain: ", DailyW[i]['rain'], "mm")
            Rain[i] = DailyW[i]['rain']
    WeatherStats = dict({'Morning': Morn, 'Day': Day, 'Evening': Eve, 'Night': Night, 'Minimum Temperature': MinTemp, 'Max Temperature':MaxTemp, 'Humidity':Humid, 'Dew Point': DewPoint,'UVI': UVIs,'Cloud Coverage(%)': Clouds, 'Rain': Rain, 'Wind Speed': WindSpeed, 'Wind Degrees': WindDeg, 'Description':Descrip}) #
    return WeatherStats
        


if __name__ == "__main__":
    main()