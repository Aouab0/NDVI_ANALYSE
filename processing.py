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
    """Génère les URLs des tuiles GEE pour les couches NDWI et NDMI."""
    roi = ee.Geometry(geo_dict)
    
    # Création d'une image médiane sans nuages sur la période choisie
    s2_median = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)) \
        .map(mask_s2_clouds) \
        .median() \
        .clip(roi) # On découpe l'image exactement sur le polygone dessiné
        
    # 1. Calcul du NDMI (Indice de Gao - Humidité Végétation & Sol profond)
    # Formule : (NIR - SWIR) / (NIR + SWIR)
    ndmi = s2_median.normalizedDifference(['B8A', 'B11']).rename('NDMI')
    vis_ndmi = {
        'min': -0.2, 'max': 0.4,
        'palette': ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
    }
    
    # 2. Calcul du NDWI (Indice de McFeeters - Eau de surface / Sols inondés)
    # Formule : (Green - NIR) / (Green + NIR)
    ndwi = s2_median.normalizedDifference(['B3', 'B8']).rename('NDWI')
    vis_ndwi = {
        'min': 0.0, 'max': 0.5,
        'palette': ['white', '#c6dbef', '#6baed6', '#2171b5', '#08306b']
    }
    
    # Extraction des URLs dynamiques pour Folium
    ndmi_url = ndmi.getMapId(vis_ndmi)['tile_fetcher'].url_format
    ndwi_url = ndwi.getMapId(vis_ndwi)['tile_fetcher'].url_format
    
    return {
        'ndmi': ndmi_url,
        'ndwi': ndwi_url
    }
