import streamlit as st
import matplotlib.pyplot as plt
from streamlit_image_select import image_select
from madrid.jupyter.madrid_function import madrid_function  
from manchester.manchester_function import manchester_function
from london.london_function import london_function
from hamburg.hamburg_function import hamburg_function
from utrecht.utrecht_function import utrecht_function
from rotterdam.rotterdam_function import rotterdam_function

# web interface
st.title('Optimal allocation of EV charging stations')
# Original repository: 

st.markdown("""
This web app permits to choose between 6 european cities and the number of EV charger stations to allocate and returns a plot of a hexagonal grid with the optimal allocation site highlighted 
""")

# utils
colNames = []
cities = ['Madrid','Manchester','London','Hamburg','Utrecht','Rotterdam']
images_dict={'Madrid':['madrid\jupyter\plots\map.png','madrid\jupyter\plots\POI.png','madrid\jupyter\plots\EV.png','madrid\jupyter\plots\Trafficpoint.png']
             ,'Manchester':['manchester\plots\map.png','manchester\plots\POI.png','manchester\plots\EV.png','manchester\plots\Trafficpoint.png']
             ,'London':['london\plots\map.png','london\plots\POI.png','london\plots\EV.png','london\plots\Trafficpoint.png']
             ,'Hamburg':['hamburg\plots\map.png','hamburg\plots\POI.png','hamburg\plots\EV.png','hamburg\plots\Trafficpoint.png']
             ,'Utrecht':['utrecht\plots\map.png','utrecht\plots\POI.png','utrecht\plots\EV.png','utrecht\plots\Trafficpoint.png']
             ,'Rotterdam':['rotterdam\plots\map.png','rotterdam\plots\POI.png','rotterdam\plots\EV.png','rotterdam\plots\Trafficpoint.png']}
fun_dict = {'Madrid':madrid_function
            ,'Manchester':manchester_function
            ,'London':london_function
            ,'Hamburg':hamburg_function
            ,'Utrecht':utrecht_function
            ,'Rotterdam':rotterdam_function}

def pred(N,city_sel):
    col_opt, poly_grid, x_lim, y_lim, gdf_roads_clip = fun_dict[city_sel](N)

    # Plotting code
    fig, ax = plt.subplots(figsize=(12, 8))
    base = gdf_roads_clip.plot(ax=ax, color='black', lw=0.4, zorder=0)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    poly_grid.plot(ax=base, facecolor=col_opt, edgecolor='black', lw=0.5, zorder=15, alpha=0.55)

    # Display the plot using Streamlit
    st.pyplot(fig)
    return

city_sel = st.selectbox('Select a city', cities, key='Cities')

img = image_select(
    label="Select a plot",
    images=images_dict[city_sel],
    captions=["Map", "POI", "EV", "Traffic"],
)

st.image(img)

with st.form("my_form"):
    N = st.slider('Number of stations', min_value=1, max_value=7, value=6, key='Number of stations')

    # submit button
    submit_button = st.form_submit_button(label='Optimize!', on_click=pred(N,city_sel))