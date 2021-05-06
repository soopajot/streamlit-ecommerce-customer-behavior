from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# eCommerce Customer Behavior

Vivamus magna justo, lacinia eget consectetur sed, convallis at tellus. Donec rutrum congue leo eget malesuada. Praesent sapien massa, convallis a pellentesque nec, egestas non nisi. Sed porttitor lectus nibh. Curabitur arcu erat, accumsan id imperdiet et, porttitor at sem. Praesent sapien massa, convallis a pellentesque nec, egestas non nisi. Vestibulum ac diam sit amet quam vehicula elementum sed sit amet dui. Cras ultricies ligula sed magna dictum porta.
"""

from PIL import Image
image = Image.open('images/streamlit.jpg')
st.image(image, caption='Notre Heatmap')

image1 = Image.open('images/img-1.png')
st.image(image1, caption='Notre img')

image2 = Image.open('images/img-2.png')
st.image(image2, caption='Notre img')

image3 = Image.open('images/img-3.png')
st.image(image3, caption='Notre img')

image4 = Image.open('images/img-4.png')
st.image(image4, caption='Notre img')

image5 = Image.open('images/img-5.png')
st.image(image5, caption='Notre img')

image6 = Image.open('images/img-6.png')
st.image(image6, caption='Notre img')

image7 = Image.open('images/img-7.png')
st.image(image7, caption='Notre img')

image8 = Image.open('images/img-8.png')
st.image(image8, caption='Notre img')

with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
