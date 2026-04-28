import ee
import pandas as pd
import streamlit as st
import requests
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

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


def create_ndvi_gif(geo_dict, year):
    """
    Génère un GIF animé du NDVI (Avril - Octobre) avec un pas de 10 jours.
    Incruste la date et une colorbar (0.15 - 0.35). FPS = 2.
    """
    roi = ee.Geometry(geo_dict)
    
    # Création des intervalles de 10 jours entre Avril et Fin Octobre
    start_date = pd.to_datetime(f'{year}-04-01')
    end_date = pd.to_datetime(f'{year}-10-31')
    date_ranges = pd.date_range(start=start_date, end=end_date, freq='10D')
    
    # Palette de couleurs (du rouge/sec au vert/dense)
    palette = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    vmin, vmax = 0.15, 0.35
    
    frames = []
    
    for i in range(len(date_ranges)-1):
        d1 = date_ranges[i].strftime('%Y-%m-%d')
        d2 = date_ranges[i+1].strftime('%Y-%m-%d')
        
        # Filtre de la collection sur les 10 jours
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(d1, d2) \
            .median()
        
        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI').clip(roi)
        
        try:
            # Récupération de l'image de GEE
            url = ndvi.getThumbURL({
                'min': vmin,
                'max': vmax,
                'palette': palette,
                'dimensions': 600, # Résolution de l'image
                'format': 'png'
            })
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            
            # --- Habillage avec Matplotlib (Titre + Colorbar) ---
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.set_title(f"Évolution NDVI : {d1} au {d2}", fontsize=14, fontweight='bold')
            ax.axis('off') # On cache les axes de coordonnées bruts
            
            # Création de la barre de légende
            cmap = LinearSegmentedColormap.from_list('ndvi', palette)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI', fontsize=12, fontweight='bold')
            
            # Sauvegarde de la figure assemblée dans la mémoire vive
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            
            frames.append(Image.open(buf))
        except Exception as e:
            # S'il y a trop de nuages ou aucune image sur ces 10 jours, on passe
            continue
            
    if not frames:
        return None
        
    # Compilation du GIF en mémoire (2 FPS = 500 ms par frame)
    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format='GIF', save_all=True, append_images=frames[1:], duration=500, loop=0)
    
    return gif_buf.getvalue()
