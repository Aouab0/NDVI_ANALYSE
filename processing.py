import ee
import streamlit as st

def init_gee():
    """Initialise la connexion à Earth Engine via Service Account"""
    credentials = ee.ServiceAccountCredentials(
        email=st.secrets["earth_engine"]["client_email"],
        key_data=st.secrets["earth_engine"]["private_key"]
    )
    ee.Initialize(credentials=credentials, project='training-462609')

def mask_s2_clouds(image):
    """Masque les nuages pixel par pixel."""
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)

def get_moisture_map_layers(geo_dict, start_date, end_date):
    """Génère les URLs des tuiles GEE pour les couches NDWI, NDMI et RGB Moyen."""
    roi = ee.Geometry(geo_dict)
    
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)) \
        .map(mask_s2_clouds)
    
    s2_median = s2_collection.median().clip(roi)
        
    # 1. Couche NDMI (Humidité)
    ndmi = s2_median.normalizedDifference(['B8A', 'B11']).rename('NDMI')
    vis_ndmi = {
        'min': -0.2, 'max': 0.4,
        'palette': ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
    }
    
    # 2. Couche NDWI (Eau de surface)
    ndwi = s2_median.normalizedDifference(['B3', 'B8']).rename('NDWI')
    vis_ndwi = {
        'min': 0.0, 'max': 0.5,
        'palette': ['white', '#c6dbef', '#6baed6', '#2171b5', '#08306b']
    }

    # 3. NOUVEAU : Couche RGB Moyen (Vraies Couleurs)
    # On utilise les bandes B4, B3, B2 avec un étirement de contraste standard (0-3000)
    vis_rgb = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
        'gamma': 1.4
    }
    
    return {
        'ndmi': ndmi.getMapId(vis_ndmi)['tile_fetcher'].url_format,
        'ndwi': ndwi.getMapId(vis_ndwi)['tile_fetcher'].url_format,
        'rgb': s2_median.getMapId(vis_rgb)['tile_fetcher'].url_format
    }
