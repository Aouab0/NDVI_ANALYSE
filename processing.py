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
    """Interroge GEE pour extraire le NDVI moyen tous les 15 jours."""
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
            scale=30,
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
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df_15j = df.resample('15D').mean().interpolate(method='linear')
        
        # Ajout des colonnes pour l'analyse statistique
        df_15j['Mois'] = df_15j.index.month
        df_15j['Année'] = df_15j.index.year
        df_15j['Saison'] = df_15j['Mois'].apply(lambda x: 'Été (Irrigation)' if x in [6, 7, 8] else 'Autre')
        return df_15j
    return pd.DataFrame()

def get_summer_ndvi_thumbs(geo_dict, start_year, end_year):
    """Génère les URL des images NDVI moyennes pour chaque été."""
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
    """Génère un GIF animé du NDVI (Avril - Octobre)."""
    roi = ee.Geometry(geo_dict)
    start_date = pd.to_datetime(f'{year}-04-01')
    end_date = pd.to_datetime(f'{year}-10-31')
    date_ranges = pd.date_range(start=start_date, end=end_date, freq='10D')
    
    palette = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    vmin, vmax = 0.15, 0.35
    frames = []
    
    for i in range(len(date_ranges)-1):
        d1 = date_ranges[i].strftime('%Y-%m-%d')
        d2 = date_ranges[i+1].strftime('%Y-%m-%d')
        
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(d1, d2) \
            .median()
        
        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI').clip(roi)
        
        try:
            url = ndvi.getThumbURL({
                'min': vmin, 'max': vmax, 'palette': palette,
                'dimensions': 600, 'format': 'png'
            })
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.set_title(f"Évolution NDVI : {d1} au {d2}", fontsize=14, fontweight='bold')
            ax.axis('off')
            
            cmap = LinearSegmentedColormap.from_list('ndvi', palette)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI', fontsize=12, fontweight='bold')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            
            frames.append(Image.open(buf))
        except Exception as e:
            continue
            
    if not frames: 
        return None
        
    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format='GIF', save_all=True, append_images=frames[1:], duration=500, loop=0)
    return gif_buf.getvalue()

def get_soil_moisture_series(geo_dict, start_date, end_date):
    """Extrait l'indice d'humidité du sol NMDI."""
    roi = ee.Geometry(geo_dict)
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
    def add_nmdi(image):
        nmdi = image.expression(
            '(GREEN - NIR) / (NIR + GREEN)', {
                'NIR': s2.select('B8A'),
                'GREEN': s2.select('B03')
            }
        ).rename('NMDI')
        
        mean_dict = nmdi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=20, 
            maxPixels=1e10
        )
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'NMDI': mean_dict.get('NMDI')
        })

    timeseries = s2.map(add_nmdi).getInfo()
    
    data = []
    for feature in timeseries['features']:
        date = feature['properties']['date']
        val = feature['properties'].get('NMDI')
        if val is not None:
            data.append({'Date': date, 'NMDI': val})
            
    df = pd.DataFrame(data)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df_15j = df.resample('15D').mean().interpolate(method='linear')
        
        df_15j['Mois'] = df_15j.index.month
        df_15j['Année'] = df_15j.index.year
        df_15j['Saison'] = df_15j['Mois'].apply(lambda x: 'Été (Irrigation)' if x in [6, 7, 8] else 'Autre')
        return df_15j
    return pd.DataFrame()

def get_summer_sm_thumbs(geo_dict, start_year, end_year):
    """Génère les URL des images NMDI pour chaque été."""
    roi = ee.Geometry(geo_dict)
    urls = {}
    vis_params = {
        'min': 0.4, 'max': 0.8,
        'palette': ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
    }
    
    for year in range(start_year, end_year + 1):
        s2_summer = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(f'{year}-06-01', f'{year}-08-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .median()
            
        nmdi = s2_summer.expression(
            '(GREEN - NIR) / (NIR + GREEN)', {
                'NIR': s2.select('B8A'),
                'GREEN': s2.select('B03')
            }
        ).rename('NMDI').clip(roi)
        
        try:
            url = nmdi.getThumbURL({
                'min': vis_params['min'], 'max': vis_params['max'],
                'palette': vis_params['palette'], 'dimensions': 400, 'format': 'png'
            })
            urls[year] = url
        except:
            urls[year] = None
            
    return urls

def create_sm_gif(geo_dict, year):
    """Génère un GIF animé de l'humidité du sol (NMDI)."""
    roi = ee.Geometry(geo_dict)
    start_date = pd.to_datetime(f'{year}-04-01')
    end_date = pd.to_datetime(f'{year}-10-31')
    date_ranges = pd.date_range(start=start_date, end=end_date, freq='10D')
    
    palette = ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
    vmin, vmax = 0.4, 0.8
    frames = []
    
    for i in range(len(date_ranges)-1):
        d1 = date_ranges[i].strftime('%Y-%m-%d')
        d2 = date_ranges[i+1].strftime('%Y-%m-%d')
        
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(d1, d2) \
            .median()
        
        nmdi = s2.expression(
            '(GREEN - NIR) / (NIR + GREEN)', {
                'NIR': s2.select('B8A'),
                'GREEN': s2.select('B03')
            }
        ).rename('NMDI').clip(roi)
        
        try:
            url = nmdi.getThumbURL({
                'min': vmin, 'max': vmax, 'palette': palette,
                'dimensions': 600, 'format': 'png'
            })
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            ax.set_title(f"Humidité du Sol NMDI : {d1} au {d2}", fontsize=14, fontweight='bold')
            ax.axis('off') 
            
            cmap = LinearSegmentedColormap.from_list('nmdi', palette)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NMDI', fontsize=12, fontweight='bold')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            
            frames.append(Image.open(buf))
        except Exception as e:
            continue
            
    if not frames: 
        return None
        
    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format='GIF', save_all=True, append_images=frames[1:], duration=500, loop=0)
    return gif_buf.getvalue()
