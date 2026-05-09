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

def mask_s2_clouds(image):
    """
    NOUVEAU : Masque les nuages pixel par pixel au lieu de rejeter l'image entière.
    Utilise la bande d'assurance qualité (QA60) de Sentinel-2.
    """
    qa = image.select('QA60')
    # Les bits 10 et 11 correspondent aux nuages et cirrus
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # On garde les pixels où les bits de nuages sont à 0 (clair)
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)

def get_soil_moisture_series(geo_dict, start_date, end_date):
    """Extrait l'indice d'humidité NDWI tous les 15 jours de manière optimisée."""
    roi = ee.Geometry(geo_dict)
    
    # On tolère jusqu'à 70% de nuages sur la scène globale, car on masque les pixels nuageux juste après
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)) \
        .map(mask_s2_clouds)
        
    def add_ndwi(image):
        # NOUVEAU : Utilisation de la fonction native de GEE (plus rapide et évite les erreurs de division par zéro)
        ndwi = image.normalizedDifference(['B8A', 'B11']).rename('NDWI')
        
        mean_dict = ndwi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=20,
            maxPixels=1e10
        )
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'NDWI': mean_dict.get('NDWI')
        })

    timeseries = s2.map(add_ndwi).getInfo()
    
    data = []
    for feature in timeseries['features']:
        date = feature['properties']['date']
        val = feature['properties'].get('NDWI')
        if val is not None:
            data.append({'Date': date, 'NDWI': val})
            
    df = pd.DataFrame(data)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date') # On s'assure que le temps est dans le bon sens
        df.set_index('Date', inplace=True)
        
        # NOUVEAU : On supprime l'interpolation aveugle. On groupe par 15 jours et on ignore les périodes sans données.
        # Le graphique sera beaucoup plus fidèle à la réalité hydrique.
        df_15j = df.resample('15D').mean().dropna()
        
        df_15j['Mois'] = df_15j.index.month
        df_15j['Année'] = df_15j.index.year
        return df_15j
    return pd.DataFrame()

def get_summer_sm_thumbs(geo_dict, start_year, end_year):
    """Génère les URL des images NDWI pour chaque été."""
    roi = ee.Geometry(geo_dict)
    urls = {}
    vis_params = {
        'min': -0.2, 'max': 0.4,
        'palette': ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
    }
    
    for year in range(start_year, end_year + 1):
        # On utilise le masque de nuages pour récupérer des pixels propres même les jours un peu nuageux
        s2_summer = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(f'{year}-06-01', f'{year}-08-31') \
            .map(mask_s2_clouds) \
            .median()
            
        ndwi = s2_summer.normalizedDifference(['B8A', 'B11']).rename('NDWI')
        
        try:
            url = ndwi.getThumbURL({
                'min': vis_params['min'], 
                'max': vis_params['max'],
                'palette': vis_params['palette'], 
                'dimensions': 400, 
                'region': roi,
                'format': 'png'
            })
            urls[year] = url
        except Exception as e:
            print(f"Année {year} impossible à générer : {e}")
            urls[year] = None
            
    return urls
