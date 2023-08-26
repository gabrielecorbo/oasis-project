import dash
from dash import dcc, html
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
import json

app = dash.Dash(__name__)

# City data
cities = [
    {"name": "Madrid", "coords": [40.4168, -3.7038], "images": ["gis.jpg", "gis.jpg", "gis.jpg", "gis.jpg"]},
    {"name": "London", "coords": [51.5074, -0.1278], "images": ["gis.jpg", "gis.jpg", "gis.jpg", "gis.jpg"]},
    {"name": "Hamburg", "coords": [53.5511, 9.9937], "images": ["gis.jpg", "gis.jpg", "gis.jpg", "gis.jpg"]}
]

app.layout = html.Div([
    dl.Map([
        dl.TileLayer(),
        dl.LayerGroup(id="markers"),
        dl.Popup(id="popup"),
        dcc.Store(id="click-data-store", data={}),
        html.Script("""
            document.addEventListener('click', function(e) {
                if (e.target && e.target.className === 'leaflet-marker-icon') {
                    var lat = e.target._latlng.lat;
                    var lon = e.target._latlng.lng;
                    var data = {lat: lat, lon: lon};
                    document.getElementById('click-data-store').value = JSON.stringify(data);
                }
            });
        """)
    ],
    center=[51.505, -0.09],
    zoom=4),
])

@app.callback(
    Output("markers", "children"),
    Output("popup", "children"),
    Input("click-data-store", "data"),
    State("markers", "children")
)
def update_map(click_data, marker_children):
    if not click_data:
        return marker_children, None
    
    lat = click_data.get('lat')
    lon = click_data.get('lon')

    popup_content = []

    if lat is not None and lon is not None:
        for city in cities:
            city_lat, city_lon = city["coords"]
            if lat == city_lat and lon == city_lon:
                popup_content.append(html.H3(city["name"]))
                for image in city["images"]:
                    image_path = f"images/{image}"
                    popup_content.append(html.Img(src=image_path, className="popup-image"))

    return marker_children, popup_content

if __name__ == '__main__':
    app.run_server(debug=True)
