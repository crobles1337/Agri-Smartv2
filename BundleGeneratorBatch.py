
from Bundle1Generatorv02 import *
from Bundle2Helpers import *
import time




def getBatch(MyCrop, xStats):
    s0 = time.time()
    #skip = ['Input1_Yakima_2019', 'Input2_Sandusky_2019', 'Input3_LaCrosse_2019', 'Input4_Fresno_2019', 'Input5_Lubbock_2019', 'Input6_WPalmBeach_2019', 'Input7_BatonRouge_2019', 'Input8_Rapides_2019', 'Input9_SouthDakota_2019', 'Input10_Fargo_2019', 'Input11_ArgosIN_2019' 'Input12_WashingtonNC_2019', 'Input13_JulesburgSD_2019', 'Input14_Navarro_2019', 'Input15_Dauphin_2019', 'Input16_PerryAL_2019', 'Input17_Nueces_2019', 'Input18_Lawton_2019'] 
    
    skip = ['Input1_Yakima_2019', 'Input2_Sandusky_2019', 'Input3_LaCrosse_2019', 'Input4_Fresno_2019', 'Input5_Lubbock_2019', 'Input6_WPalmBeach_2019', 'Input7_BatonRouge_2019', 'Input8_Rapides_2019', 'Input9_SouthDakota_2019', 'Input10_Fargo_2019', 'Input11_ArgosIN_2019', 'Input12_WashingtonNC_2019', 'Input13_JulesburgSD_2019', 'Input14_Navarro_2019', 'Input15_Dauphin_2019'] # Input1_Yakima_2019 is an example
    pre = 'SentImages_Training\Imagery' 

    #inputname = 'Input1_Yakima_2019'
    for dirp in os.listdir(pre):
        s1 = time.time()
        t = s1 - s0
        print("starting", dirp, t)
        if 'Input' in dirp:
            if not dirp in skip:
                print("now", dirp)
                inputname = dirp
                BundleGenerate(inputname, Crop=MyCrop, xStats=xStats)
                
#            ipre = os.path.join(pre, dirp)
#            ip1 = os.path.join(ipre, 'Imagery')
#            iext = os.path.join('Imagery', os.listdir(ip1)[0]) #saves only ext
#            ip2 = os.path.join(ip1, os.listdir(ip1)[0])


#            for b in os.listdir(ip2):
#                if 'B04' in b:
#                    iext = os.path.join(iext, b)
            
getBatch('Wheat', xStats=True) 