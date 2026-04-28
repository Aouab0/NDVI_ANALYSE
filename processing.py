import ee
import pandas as pd
import streamlit as st
def init_gee():
        """Initialise la connexion à Earth Engine"""
        credentials = ee.ServiceAccountCredentials(
            email=st.secrets["earth_engine"]["client_email"],
            key_data=st.secrets["earth_engine"]["private_key"]
        )
        ee.Initialize(credentials=credentials, project='training-462609')

def get_ndvi_series(geojson_polygons, start_date, end_date):
    """
    Interroge GEE pour extraire le NDVI moyen tous les 15 jours sur une zone.
    """
    # Création de la géométrie GEE à partir des polygones locaux
    roi = ee.Geometry.MultiPolygon(geojson_polygons)
    
    # Sentinel-2 harmonisé (Surface Reflectance)
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
    def add_ndvi(image):
        # Calcul du NDVI: (B8 - B4) / (B8 + B4)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        # Réduction (moyenne) sur notre zone d'intérêt
        # Scale à 100m (au lieu de 10m) pour éviter les timeouts serveur sur de grandes zones
        mean_dict = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=100, 
            maxPixels=1e10
        )
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'NDVI': mean_dict.get('NDVI')
        })

    # Extraction des données via map()
    timeseries = s2.map(add_ndvi).getInfo()
    
    # Transformation en DataFrame Pandas
    data = []
    for feature in timeseries['features']:
        date = feature['properties']['date']
        ndvi_val = feature['properties'].get('NDVI')
        if ndvi_val is not None:
            data.append({'Date': date, 'NDVI': ndvi_val})
            
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Lissage et regroupement par quinzaine (15 jours) pour nettoyer les nuages résiduels
    df_15j = df.resample('15D').mean().interpolate(method='linear')
    return df_15j
