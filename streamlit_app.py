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

add_selectbox = st.sidebar.selectbox("You Can do the following using this Website ",
(“Data Pre Processing using Pandas”, “Correcting”,‘Completing’,
‘Creating’,‘Modeling using Sklearn’))
