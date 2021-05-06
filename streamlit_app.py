from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pickle as pkle
import os.path
"""
# eCommerce Customer Behavior

Vivamus magna justo, lacinia eget consectetur sed, convallis at tellus. Donec rutrum congue leo eget malesuada. Praesent sapien massa, convallis a pellentesque nec, egestas non nisi. Sed porttitor lectus nibh. Curabitur arcu erat, accumsan id imperdiet et, porttitor at sem. Praesent sapien massa, convallis a pellentesque nec, egestas non nisi. Vestibulum ac diam sit amet quam vehicula elementum sed sit amet dui. Cras ultricies ligula sed magna dictum porta.
"""

# create a button in the side bar that will move to the next page/radio button choice
next = st.sidebar.button('Next on list')

# will use this list and next button to increment page, MUST BE in the SAME order
# as the list passed to the radio button
new_choice = ['Transactions','Items','Visitors']

# This is what makes this work, check directory for a pickled file that contains
# the index of the page you want displayed, if it exists, then you pick up where the
#previous run through of your Streamlit Script left off,
# if it's the first go it's just set to 0
if os.path.isfile('next.p'):
    next_clicked = pkle.load(open('next.p', 'rb'))
    # check if you are at the end of the list of pages
    if next_clicked == len(new_choice):
        next_clicked = 0 # go back to the beginning i.e. homepage
else:
    next_clicked = 0 #the start

# this is the second tricky bit, check to see if the person has clicked the
# next button and increment our index tracker (next_clicked)
if next:
    #increment value to get to the next page
    next_clicked = next_clicked +1

    # check if you are at the end of the list of pages again
    if next_clicked == len(new_choice):
        next_clicked = 0 # go back to the beginning i.e. homepage

# create your radio button with the index that we loaded
choice = st.sidebar.radio("go to",('Transactions','Items', 'Visitors'), index=next_clicked)

# pickle the index associated with the value, to keep track if the radio button has been used
pkle.dump(new_choice.index(choice), open('next.p', 'wb'))

# finally get to whats on each page
if choice == 'Transactions':
    st.header('this is Transactions')
elif choice == 'Items':
    st.header('here is a Items page')
elif choice == 'Visitors':
    st.header('A Visitors of some sort')


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
