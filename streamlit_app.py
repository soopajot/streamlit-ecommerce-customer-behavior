from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pickle as pkle
import os.path
from PIL import Image

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
"""
# eCommerce Customer Behavior

Vivamus magna justo, lacinia eget consectetur sed, convallis at tellus. Donec rutrum congue leo eget malesuada. Praesent sapien massa, convallis a pellentesque nec, egestas non nisi. Sed porttitor lectus nibh. Curabitur arcu erat, accumsan id imperdiet et, porttitor at sem. Praesent sapien massa, convallis a pellentesque nec, egestas non nisi. Vestibulum ac diam sit amet quam vehicula elementum sed sit amet dui. Cras ultricies ligula sed magna dictum porta. 
"""

# will use this list and next button to increment page, MUST BE in the SAME order
# as the list passed to the radio button
new_choice = ['Home','Visitors','Transactions', 'Items']

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
choice = st.sidebar.radio("go to",('Home','Visitors','Transactions', 'Items'), index=next_clicked)

# pickle the index associated with the value, to keep track if the radio button has been used
pkle.dump(new_choice.index(choice), open('next.p', 'wb'))

# finally get to whats on each page
if choice == 'Home':
    #SHOW IMAGES
    image = Image.open('images/streamlit.jpg')
    st.image(image, caption='E-commerce Customer Behavior')
elif choice == 'Visitors':
    st.header('Dataset Visitors')
    # DATASET STATS
    stats = pd.read_csv('csv/stats.csv')
    st.write("Voici un aperçu du dataset")
    st.write(stats)
    #SHOW IMAGES
    image1 = Image.open('images/img-1.png')
    st.image(image1, caption='Nombre de visiteurs')
########### TIME TRANSACTIONS #######
elif choice == 'Transactions':
    st.header('Dataset Transactions-Time')
    time_sum_tran_sample = pd.read_csv('csv/time_sum_tran_sample.csv')
    st.write(time_sum_tran_sample)

    #time_sum_tran_sample_1 = pd.read_csv('csv/time_sum_tran_sample_1.csv')
    #st.write("Le temps total par transaction")
    #st.line_chart(time_sum_tran_sample_1)

    st.header("Le temps nécessaire pour déclencer une transaction")

    # Temps de Transactions moins d'une heure
    sum_tran_1h = time_sum_tran_sample.loc[round(time_sum_tran_sample['sum_time_minute']) <= 60]

    # Temps de transactions moins de 10 minutes
    sum_trans_10min = sum_tran_1h.loc[round(sum_tran_1h['sum_time_minute']) <= 10]

    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Le temps nécessaire pour déclencer une transaction')
    sns.histplot(time_sum_tran_sample['sum_time_hour'], bins=24, kde=True, color='orange', ax=ax1)
    ax1.set_title("heure")
    sns.histplot(sum_tran_1h['sum_time_minute'], bins=6, kde=True, color='red', ax=ax2)
    ax2.set_title("en moins de 1 heure")
    sns.histplot(sum_trans_10min['sum_time_minute'], bins=6, kde=True, ax=ax3)
    ax3.set_title("en moins de 10 minutes")
    st.pyplot(fig)


    st.header("Clustering sur Transactions")
    fig, ax = plt.subplots()
    k1 = KMeans(n_clusters=50).fit(time_sum_tran_sample)
    Z = linkage(k1.cluster_centers_, method='ward', metric='euclidean')
    dendrogram(Z, leaf_rotation=70.)
    st.pyplot(fig)

    # AGGLOMERATIVECLUSTERING PAR 3
    ac = AgglomerativeClustering(n_clusters = 3).fit(k1.cluster_centers_)
    cd = pd.DataFrame(k1.cluster_centers_)

    time_sum_tran_sample['kmean1_label'] = k1.labels_

    # Ajouter la colonne agglo_label
    for i in list(cd.index):
        time_sum_tran_sample.loc[time_sum_tran_sample['kmean1_label'] == cd.index[i], 'agglo_label'] = ac.labels_[i]

    # Get new centroids = mean of 3 labels from Agglo
    new_centroids = time_sum_tran_sample.groupby('agglo_label').mean()
    new_time_sum_tran_sample = time_sum_tran_sample.drop(['agglo_label'], axis=1)

    k2 = KMeans(n_clusters=3, init=new_centroids)
    k2.fit(new_time_sum_tran_sample)

    # Centroids and labels
    k2_centroids = k2.cluster_centers_
    k2_labels = k2.labels_

    # Ajouter la colonne kmean2_label
    time_sum_tran_sample['kmean2_label'] = k2.labels_
    #time_sum_tran_sample

    #sum_tran_1h_1 = pd.read_csv('csv/sum_tran_1h_1.csv')
    #st.write("Le temps total par transaction 1h")
    #st.line_chart(sum_tran_1h_1)

    #sum_trans_10min_1 = pd.read_csv('csv/sum_trans_10min_1.csv')
    #st.write("Le temps total par transaction 10min")
    #st.bar_chart(sum_trans_10min_1)
    
    #SHOW IMAGES
    #image3 = Image.open('images/img-3.png')
    #st.image(image3, caption='Notre img')
elif choice == 'Items':
    st.header('Dataset Items page')
    # DATASET ITEMS
    items = pd.read_csv('csv/items.csv')
    st.write("Voici un aperçu du dataset")
    st.write(items)
    #SHOW IMAGES
    image2 = Image.open('images/img-2.png')
    st.image(image2, caption='Items plus achetés')



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





#stats_all = pd.read_csv('csv/stats_all.csv')
#top_produits_merged_buy_all = pd.read_csv('csv/top_produits_merged_buy_all.csv')


#st.write(stats_all)
#st.write(top_produits_merged_buy_all)