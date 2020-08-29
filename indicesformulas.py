import numpy as np



def getNDVI(B4Red, B8NIR):
    "Normalized Difference Vegetation Index used as the primary vegetation health index." 
    #temp  below
    NDVI = np.where(
        (B8NIR+B4Red)==0., 
        0,
        (B8NIR-B4Red)/(B8NIR+B4Red)
    )   
    
    return NDVI

def getNDRE(B5RE, B8NIR):
    "Noramlized Difference Red Edge used for chlorophyll content, less sensitive to top-layer canopy, measures closer to soil."
    NDRE = np.where(
        (B8NIR+B5RE)==0.,
        0.,
        (B8NIR-B5RE)/(B8NIR+B5RE)
    ) 
    return NDRE

def getRECI(B5RE, B8NIR):
    "Red Edge Chlorophyll Index is a useful general measurement of crop stress (higher values indicate less stress)"
    RECI = np.where(
        B5RE==0.,
        0.,
        (B8NIR/B5RE) - 1
    )
    return RECI

def getCCCI(NDVI, NDRE):
    "Canopy Chlorophyll Content Index used for low-sensitivity measurements of dense vegetation."
    newCCCI = np.where(
        NDVI==0.0,
        0.,
        NDRE/NDVI
    )
    return newCCCI

def getSAVI(B4Red, B8NIR):
    "Soil-Adjusted Vegetation Index is a vegetation index best for early stage crops."
    L = 0 #L is variable from -1 to 1. For high green vegetation, L is set to 0, whereas for low green vegetation, it is set to 1.
    SAVI = np.where(
        ((B8NIR+B4Red+L) * (1+L)) ==0.,
        0.,
        (B8NIR - B4Red) / ((B8NIR+B4Red+L) * (1+L))
    )
    return SAVI

def getSIPI(B2Blue, B4Red, B8NIR):
    "Structure Insensitive Pigment Index used as a stress indicator. High values indicate plant disease/stress."
    SIPI = np.where(
        (B8NIR-B4Red)==0.,
        0.,
        (B8NIR-B2Blue)/(B8NIR-B4Red)
    )
    return SIPI

def getARVI(B2Blue, B4Red, B8NIR):
    "Atmospherically-Resistant Vegetation Index is a vegetation health index corrected for atmospheric effects such as pollution."
    ARVI = np.where(
        (B8NIR+(2*B4Red)+B2Blue)==0.,
        0.,
        (B8NIR-(2*B4Red)+B2Blue)/(B8NIR+(2*B4Red)+B2Blue)
    )
    return ARVI

def getGVMI(B8NIR, B11SWIR1):
    "Global Vegetation Moisture Index used as a rough indicator for vegetation water content."
    GVMI = np.where(
        ((B8NIR+0.1)+(B11SWIR1+0.2))==0.,
        0.,
        (B8NIR+0.1)-(B11SWIR1+0.2)/(B8NIR+0.1)+(B11SWIR1+0.2)
    )
    return GVMI

def getNDMI(B8ANNIR, B11SWIR1, B12SWIR2):
    "Normalized Difference Moisture Index used as rough index for vegetation water content."
    NDMI = np.where(
        (B8ANNIR + (B11SWIR1-B12SWIR2))==0.,
        0.,
        (B8ANNIR - (B11SWIR1-B12SWIR2)) / (B8ANNIR + (B11SWIR1-B12SWIR2))
    )
    
    NDMI = np.where(
        NDMI>1.,
        0.,
        NDMI
    )
    
    NDMI = np.where(
        NDMI<-1.,
        0.,
        NDMI
    )
    return NDMI

def getGCI(B3Green, B8NIR):
    "Green Chlorophyll Index used as a general stress and chlorophyll index, with higher values indicating less stress."
    GCI = np.where(
        B3Green==0.,
        0.,
        B8NIR/B3Green
    )
    return GCI

def getNDWI(B8ANNIR, B11SWIR1):
    "Normalized Difference Water Index used as rough index for vegetation water content. "
    NDWI = np.where(
        (B11SWIR1+B8ANNIR)==0.,
        0.,
        (B8ANNIR - B11SWIR1)/(B11SWIR1+B8ANNIR)
    )
    return NDWI

def getMI(B8ANNIR, B12SWIR2):
    "Moisture index used for moisture map as a rough index for index for vegetation moisture."
    MI = np.where(
        (B8ANNIR+B12SWIR2)==0.,
        0.,
        (B8ANNIR-B12SWIR2)/(B8ANNIR+B12SWIR2)
    )
    return MI

