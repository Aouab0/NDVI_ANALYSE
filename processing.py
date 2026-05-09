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

def get_soil_moisture_series(geo_dict, start_date, end_date):
    """Extrait l'indice d'humidité (NDWI/NMDI) tous les 15 jours."""
    roi = ee.Geometry(geo_dict)
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
    def add_ndwi(image):
        # Indice exploitant l'infrarouge (NIR et SWIR) pour l'humidité
        ndwi = image.expression(
            '(NIR - SWIR) / (NIR + SWIR)', {
                'NIR': image.select('B8A'),
                'SWIR': image.select('B11')
            }
        ).rename('NDWI')
        
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
        df.set_index('Date', inplace=True)
        # Lissage par quinzaine
        df_15j = df.resample('15D').mean().interpolate(method='linear')
        
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
        # On filtre la collection pour l'été de l'année en cours
        s2_summer = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(f'{year}-06-01', f'{year}-08-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .median()
            
        # Calcul de l'indice NDWI
        ndwi = s2_summer.expression(
            '(NIR - SWIR) / (NIR + SWIR)', {
                'NIR': s2_summer.select('B8A'),
                'SWIR': s2_summer.select('B11')
            }
        ).rename('NDWI')
        
        try:
            # L'ajout du paramètre 'region': roi est OBLIGATOIRE ici pour que GEE accepte de créer l'image
            url = ndwi.getThumbURL({
                'min': vis_params['min'], 
                'max': vis_params['max'],
                'palette': vis_params['palette'], 
                'dimensions': 400, 
                'region': roi,  # <-- C'est ce paramètre qui débloque l'affichage
                'format': 'png'
            })
            urls[year] = url
        except Exception as e:
            # Affichage de l'erreur dans la console pour faciliter le débogage si besoin
            print(f"Erreur pour l'année {year} : {e}")
            urls[year] = None
            
    return urls
