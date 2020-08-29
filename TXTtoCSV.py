import pandas as pd


"CODE USED TO create COMMAS"
#with open('TextFiles\Stationswcommas.txt', 'r') as file:
#    data = file.read().replace(',,', ',')

#file1 = open('TextFiles\Stationswcommas.txt', "w")
#file1.write(data)
#file1.close()
"CODE USED TO create COMMAS"

read_file = pd.read_csv(r'TextFiles\Stationswcommas.txt')
read_file.to_csv(r'CSVFiles\WeatherStations.csv', index=None)
