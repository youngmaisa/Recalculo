import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import traceback  
import io


# CONFIGURACION INICIAL  ----------------------------------------------------------------

img = Image.open('entel.jpg')
st.set_page_config(page_title='Recalculo Entel', page_icon=img, layout='wide')


