from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import geopandas
import shapely
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from rasterio.windows import Window
#%matplotlib inline



"Part 1: Accessing copernicus query data and SentinelAPI and downloading images"
#connect to API
api = SentinelAPI('croblitos', 'LucklessMonkey$30', 'https://scihub.copernicus.eu/dhus')

#search

#FOCUS ON THIS FOOTPRINT PART!!!
#footprint = geojson_to_wkt()
"If i can figure out how to convert sections with wheat to a geojson, that would help"
print("STARTEDNEW")
#MRandCoord = '37.51634, -120.69970' # should work as a Well Known Text object
#my_geoJSON = '{ "type": "Point", "coordinates": ['37.248037119' ,'-120.99972356'] }'
#footprint = 'POINT(37.2480 -120.9997)'


"SOMEHOW THE BELOW CODE WORKS, I HAVE TO DO THE BACKWARDS THING, WGS84"
#footprint = 'POLYGON((0 0,1 1,0 1,0 0))'
footprint = 'POINT(-102.16 34.33)'
#footprint = 'footprint: "intersects(POINT(37.2480 -120.9997))"'
#returns product matches by product ID
products = api.query(footprint,
                    date=('20190601', date(2019, 6, 20)),
                    area_relation='Intersects',
                    platformname = 'Sentinel-2',
                    cloudcoverpercentage=(0, 10)

)
"SOMEHOW THE ABOVE CODE WORKS"


"PRODUCTS WILL RETURN : dict[string, dict] Products returned by the query as a dictionary with the product ID as the key and the product's attributes (a dictionary) as the value."
print(len(products)) #number of products in search result
print(products) #where bulk info is stored
print('geojson')
print(api.to_geojson(products)) #same info but in geojson format
print('geopanda')
print(api.to_geodataframe(products)) #in dataframe format
apigeodf = api.to_geodataframe(products)
print(apigeodf['uuid'][0]) #code for accesing id

print('proddata')

# Save this cuz its okay'997a50f4-10cb-4ed9-9343-3f977fefe4d1'
print(api.get_product_odata('1d6bf68e-0f1f-4662-8b83-9d084a62ff57')) #full gives similar info to products print
#UUID = PRODUCTID "4 e225793-e5ca-408f-bfc5-7b64bc68473f"
print("prodid came before this")
#don't forget to comment out download once this is over
#api.download('0363e78c-fb72-4a6a-94e2-39ae700e077b') #downloads into same folder as app.py as () file, started at 10:54 am, finished at...




"this download should be ready in 24 hours"
"Dealing with Long Term Archive/LTA"
###api.download('32199ff8-b5d5-4c80-9b93-175ed4916355') #download Merced region image. 
###product_info = api.get_product_odata('32199ff8-b5d5-4c80-9b93-175ed4916355')
###3if product_info['Online']:
###    print('Product {} is online. Starting download.'.format('32199ff8-b5d5-4c80-9b93-175ed4916355'))
###    api.download('32199ff8-b5d5-4c80-9b93-175ed4916355')
###else:
###    print('Product {} is not online.'.format('32199ff8-b5d5-4c80-9b93-175ed4916355'))



#PROVIDES 1 OF THESE PER PRODUCT --- {'title': 'S2B_MSIL1C_20180904T184019_N0206_R070_T11SKB_20180904T233822', 'link': "https://scihub.copernicus.eu/dhus/odata/v1/Products('32199ff8-b5d5-4c80-9b93-175ed4916355')/$value", 'link_alternative': "https://scihub.copernicus.eu/dhus/odata/v1/Products('32199ff8-b5d5-4c80-9b93-175ed4916355')/", 'link_icon': "https://scihub.copernicus.eu/dhus/odata/v1/Products('32199ff8-b5d5-4c80-9b93-175ed4916355')/Products('Quicklook')/$value", 'summary': 'Date: 2018-09-04T18:40:19.024Z, Instrument: MSI, Mode: , Satellite: Sentinel-2, Size: 802.12 MB', 'datatakesensingstart': datetime.datetime(2018, 9, 4, 18, 40, 19, 24000), 'beginposition': datetime.datetime(2018, 9, 4, 18, 40, 19, 24000), 'endposition': datetime.datetime(2018, 9, 4, 18, 40, 19, 24000), 'ingestiondate': datetime.datetime(2018, 9, 5, 2, 20, 59, 699000), 'orbitnumber': 7818, 'relativeorbitnumber': 70, 'cloudcoverpercentage': 6.708600000000001, 'sensoroperationalmode': 'INS-NOBS', 'gmlfootprint': '<gml:Polygon srsName=http://www.opengis.net/gml/srs/epsg.xml#4326 xmlns:gml=http://www.opengis.net/gml>\n   <gml:outerBoundaryIs>\n      <gml:LinearRing>\n         <gml:coordinates>37.8980997878049,-120.41197 37.92768043331917,-119.16434 36.93866731150362,-119.13603 36.910119161555244,-120.36742 37.8980997878049,-120.41197</gml:coordinates>\n      </gml:LinearRing>\n   </gml:outerBoundaryIs>\n</gml:Polygon>', 'footprint': 'MULTIPOLYGON (((-120.41197 37.8980997878049, -120.36742 36.910119161555244, -119.13603 36.93866731150362, -119.16434 37.92768043331917, -120.41197 37.8980997878049)))', 'tileid': '11SKB', 'hv_order_tileid': 'SB11K', 'format': 'SAFE', 'processingbaseline': '02.06', 'platformname': 'Sentinel-2', 'filename': 'S2B_MSIL1C_20180904T184019_N0206_R070_T11SKB_20180904T233822.SAFE', 'instrumentname': 'Multi-Spectral Instrument', 'instrumentshortname': 'MSI', 'size': '802.12 MB', 's2datatakeid': 'GS2B_20180904T184019_007818_N02.06', 'producttype': 'S2MSI1C', 'platformidentifier': '2017-013A', 'orbitdirection': 'DESCENDING', 'platformserialidentifier': 'Sentinel-2B', 'processinglevel': 'Level-1C', 'identifier': 'S2B_MSIL1C_20180904T184019_N0206_R070_T11SKB_20180904T233822', 'uuid': '32199ff8-b5d5-4c80-9b93-175ed4916355', 'level1cpdiidentifier': 'S2B_OPER_MSI_L1C_TL_EPAE_20180904T233822_A007818_T11SKB_N02.06', 'granuleidentifier': 'S2B_OPER_MSI_L1C_TL_EPAE_20180904T233822_A007818_T11SKB_N02.06', 'datastripidentifier': 'S2B_OPER_MSI_L1C_DS_EPAE_20180904T233822_S20180904T185252_N02.06'}), ('05021f5a-bbda-484c-a904-9ac41927af6f', "
"Each image is approximately 12039 sq. km or 2975123 acres, or 1203989 hectares, or 4648 sq. km"







