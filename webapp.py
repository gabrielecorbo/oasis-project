import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_select import image_select
from madrid.jupyter.optimization import gen_sets  

# web interface
st.title('Optimal allocation of EV charging stations')

st.markdown("""
Original repository: 

This web app ... 
""")

# utils
colNames = []
cities = ['Madrid','Manchester','London','Hamburg','Utrecht','Rotterdam']
images_dict={'Madrid':['2-madrid\jupyter\plots\map.png','2-madrid\jupyter\plots\POI.png','2-madrid\jupyter\plots\EV.png','2-madrid\jupyter\plots\Trafficpoint.png'],
             'Manchester':['1-manchester\plots\map.png','1-manchester\plots\POI.png','1-manchester\plots\EV.png','1-manchester\plots\Trafficpoint.png']
             ,'London':['5-london\plots\map.png','5-london\plots\POI.png','5-london\plots\EV.png','5-london\plots\Trafficpoint.png']
             ,'Hamburg':['4-hamburg\plots\map.png','4-hamburg\plots\POI.png','4-hamburg\plots\EV.png','4-hamburg\plots\Trafficpoint.png']
             ,'Utrecht':['6-utrecht\plots\map.png','6-utrecht\plots\POI.png','6-utrecht\plots\EV.png','6-utrecht\plots\Trafficpoint.png']
             ,'Rotterdam':['3-rotterdam\plots\map.png','3-rotterdam\plots\POI.png','3-rotterdam\plots\EV.png','3-rotterdam\plots\Trafficpoint.png']}

#def pred(N):

city_sel = st.selectbox('City', cities, key='Cities')

img = image_select(
    label="Select a plot",
    images=images_dict[city_sel],
    captions=["Map", "POI", "EV", "Traffic"],
)

st.image(img)

N = st.slider('Number of stations', min_value=1, max_value=7, key='Number of stations')

# submit button
#submit_button = st.form_submit_button(label='Calculate!', on_click=pred(N))