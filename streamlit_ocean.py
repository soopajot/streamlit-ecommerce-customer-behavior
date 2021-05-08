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
# Projet & Résultats

""" 

st.header("Introduction")

""""
Le Dataset de notre projet provient du site Kaggle https://www.kaggle.com/retailrocket/ecommerce-dataset, dans ce jeu de données nous avons les données comportementales des utilisateurs : **visitorid**, **events**, **timestamps**, **transactionid** et **itemid**.     
L’objectif de ce projet est d’analyser le **comportement** des **acheteurs** d’un site e-commerce.
Comprendre les habitudes de consommation de nos acheteurs afin d’observer des **similitudes** et des **disparités** entre les différents **groupes**.
Notre problématique est que nous avons peu de variables et avons du **explorer** la variable **timestamp**.
L'enjeu est d'analyser les différents groupe d’acheteurs via un modèle de **clustering**. 

""" 

# DATASET DF_SAMPLE
df_sample = pd.read_csv('csv/df_sample.csv')
df_sample = df_sample.fillna(0)
st.write("Voici un aperçu du dataset: ", "Nous avons pris le soin de ramplacer les **NaN** par des **zéros** afin de ne perdre aucune informations")
st.write(df_sample)

"""
# Premières analyses & Dataviz 

Regardons la composition des évènements qui compose notre Dataset.

"""

fig=plt.figure(figsize=(4,4))

plt.pie(df_sample.event.value_counts(),
        labels=['view', 'addtocart','transaction'],
        colors=['steelblue','orange','green'],
        explode = [0.1, 0, 0],
        autopct=lambda x : str(round(x, 2)) + '%',
        pctdistance=0.7, labeldistance=1.3,
        shadow=True)
plt.legend()
st.pyplot(fig);
st.write("Ce 1er graphique permet de comprendre que l’essentiel de notre dataset est constitué de **vues 97%** et **2,3%** **d'ajouts au panier**. A contrario, l’évènement « **transaction** » ne représente même pas **1%**. Cette analyse nous pousse à nous intéresser au comportement des **visiteurs** ainsi que les **acheteurs**. (voir graph ci-dessous)")

###### Premières dataviz sur le comportement des visiteurs ##### 
         
# Style 
sns.set_theme()
sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette(['#39A7D0','#36ADA4'])

# Format 
fig_1=plt.figure(figsize = (14,14))

# Visiteurs par mois
plt.subplot(221)
sns.countplot(x='month', data=df_sample)
plt.title('Nombre de visiteurs par mois', fontsize=15)
plt.xlabel("Mois",fontsize=15)
plt.ylabel("Nombre de visites",fontsize=15)
plt.grid()

# Visiteurs par jour
plt.subplot(222)
sns.countplot(x='day', data=df_sample)
plt.title('Nombre de visiteurs par jour', fontsize=15)
plt.xlabel("Jours",fontsize=15)
plt.ylabel("Nombre de visites",fontsize=15)
plt.grid()

st.pyplot(fig_1);

st.write("D'après ces deux histogrammes, nous avons remarqué que les consommateurs avaient effectué le plus d’actions au mois de **juillet** (quid promotion ?).")

fig_2=plt.figure(figsize = (14,14))

#Visiteurs par jour de semaine
plt.subplot(221)
sns.countplot(x='dayofweek', data=df_sample)
plt.title('Nombre de visiteurs par jour de semaine', fontsize=15)
plt.xlabel("Jours de la semaine",fontsize=15)
plt.ylabel("Nombre de visites",fontsize=15)
plt.grid()

# Visiteurs par heure
plt.subplot(222)
sns.countplot(x='hour', data=df_sample)
plt.title('Nombre de visiteurs par heure', fontsize=15)
plt.xlabel("Heures",fontsize=15)
plt.ylabel("Nombre de visites",fontsize=15)
plt.grid()

st.pyplot(fig_2);
         
st.write("De plus les visiteurs ont tendance à **diminuer leurs activités** sur le site **durant les week-ends** et sont **réactifs en semaine**. Au niveau des horaires les visiteurs effectuent leurs actions à partir de **17 heures jusqu’au lendemain vers 7 heures**. Nous allons maintenant étudier la relation entre plusieurs variables.")

"""
# Relation entre les différentes variables

"""
st.write("Le heatmap nous permet d’analyser les corrélations entre les variables **nb_visites**, **nb_views**, **nb_addtocarts** et **nb_transactions**. Nous pouvons voir que toutes les variables sont corrélées (**0,76 à 1**). Ainsi, le **nombre de visites** et le **nombre de vues** sont **parfaitement corrélées**, ce qui semble logique car une visite est accompagnée quasi systématiquement d’une vue."
         
"On remarque également la **forte corrélation** entre **le nombre d'ajouts au panier** et **le nombre de transactions** (**0.9**)." "Effectivement un produit ajouté au panier a bien plus de chance d’être acheté."

"Ce heatmap présente également les variables liées au temps (**voir dans la partie présentation des datasets**) elle montre clairement que les variables citées ci-dessus ne sont pas corrélées avec le **temps moyen passé par un visiteur pour effectuer une transaction** (très **proche de 0**), et **faiblement corrélés** avec **le temps total passé par un visiteur pour effectuer ses achats** (**entre 0,30 et 0,34**).")

# DATASET Stats pour la matrice de corrélation
stats_sample = pd.read_csv('csv/stats_sample.csv')

fig_3=plt.figure(figsize=(8,5))
sns.heatmap(stats_sample.corr(), annot=True, cmap='RdBu_r', center=0)
st.pyplot(fig_3);


"""
# Analyse du temps nécessaire pour déclencher une transaction

Comme vu dans l'introduction nous avons exploité la variable **“timestamp”**, pour définir une **visite convertissante**, nous avons décidé de **calculer le temps passé pour déclencher la transaction dans une limite de 24h**. Il s’agit dans un premier temps de calculer la **différence** entre **l’heure à laquelle la transaction s’est produite** et **l’heure à laquelle chaque événement s’est produit**.

"""

time_sum_tran_sample = pd.read_csv('csv/time_sum_tran_sample.csv')

# Temps de Transactions moins d'une heure
sum_tran_1h = time_sum_tran_sample.loc[round(time_sum_tran_sample['sum_time_minute']) <= 60]

# Temps de transactions moins de 10 minutes
sum_trans_10min = sum_tran_1h.loc[round(sum_tran_1h['sum_time_minute']) <= 10]

fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
#fig.suptitle('Le temps nécessaire pour déclencer une transaction', fontsize=15)

sns.histplot(time_sum_tran_sample['sum_time_hour'], bins=24, kde=True, color='orange', ax=ax1)
ax1.set_title("Dans l'heure", fontsize=15)
sns.histplot(sum_tran_1h['sum_time_minute'], bins=6, kde=True, color='red', ax=ax2)
ax2.set_title("En moins de 1 heure", fontsize=15)
sns.histplot(sum_trans_10min['sum_time_minute'], bins=6, kde=True, ax=ax3)
ax3.set_title("En moins de 10 minutes", fontsize=15)
st.pyplot(fig);

st.write(" Nous observons un comportement majoritaire de la part des visiteurs : ils  prennent moins d’une heure à effectuer un achat."
"On constate aussi qu’une grande partie de nos visiteurs effectuent leurs achats en moins de 10 minutes. Partant de ce constat nous avons décidé d'observer l'achat en moins de 10 minutes, les acheteurs leurs achètent entre 1 et 3 minutes environ.")
