import ee
import pandas as pd
import streamlit as st

def init_gee():
    """Initialise la connexion à Earth Engine via Service Account"""
    credentials = ee.ServiceAccountCredentials(
        email=st.secrets["earth_engine"]["client_email"],
        key_data=st.secrets["earth_engine"]["private_key"]
    )
    ee.Initialize(credentials=credentials, project='training-462609')

def get_ndvi_series(geo_dict, start_date, end_date):
    """
    Interroge GEE pour extraire le NDVI moyen tous les 15 jours.
    """
    # GEE détecte automatiquement si c'est un Polygon ou MultiPolygon via le dict GeoJSON
    roi = ee.Geometry(geo_dict) 
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        mean_dict = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=30, # Échelle à 30m pour équilibrer précision et temps de calcul
            maxPixels=1e10
        )
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'NDVI': mean_dict.get('NDVI')
        })

    timeseries = s2.map(add_ndvi).getInfo()
    
    data = []
    for feature in timeseries['features']:
        date = feature['properties']['date']
        ndvi_val = feature['properties'].get('NDVI')
        if ndvi_val is not None:
            data.append({'Date': date, 'NDVI': ndvi_val})
            
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df_15j = df.resample('15D').mean().interpolate(method='linear')
    
    # Ajout des colonnes pour l'analyse statistique du PFE
    df_15j['Mois'] = df_15j.index.month
    df_15j['Année'] = df_15j.index.year
    df_15j['Saison'] = df_15j['Mois'].apply(lambda x: 'Été (Irrigation)' if x in [6, 7, 8] else 'Autre')
    
    return df_15j

def get_summer_ndvi_thumbs(geo_dict, start_year, end_year):
    """
    Génère les URL des images NDVI moyennes (Juin à Août) pour chaque année.
    """
    roi = ee.Geometry(geo_dict)
    urls = {}
    
    vis_params = {
        'min': 0.0, 'max': 0.8,
        'palette': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    }
    
    for year in range(start_year, end_year + 1):
        s2_summer = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(f'{year}-06-01', f'{year}-08-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .median()
            
        ndvi = s2_summer.normalizedDifference(['B8', 'B4']).rename('NDVI').clip(roi)
        
        try:
            url = ndvi.getThumbURL({
                'min': vis_params['min'], 'max': vis_params['max'],
                'palette': vis_params['palette'], 'dimensions': 400, 'format': 'png'
            })
            urls[year] = url
        except:
            urls[year] = None
            
    return urls
