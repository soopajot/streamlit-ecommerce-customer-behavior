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
import time



st.sidebar.header("eCommerce Customer Behavior")
st.sidebar.subheader("Menu")

menu = st.sidebar.radio("Affichez",('Home', 'Projet et résultats', 'Présentation des 3 dataset'))
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.info('Réalisée par Mélissa Jaffal, Océane Hung May, Sadali Hewa, Sooyoung Lee Pajot')
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
image = Image.open('images/datascientest.png')
st.sidebar.image(image, width=200)

if menu == 'Home':
    """
    # eCommerce Customer Behavior
    """
    image = Image.open('images/streamlit.jpg') #nom image streamlit
    st.image(image, width=698)
elif menu == "Projet et résultats":
    """
    # Projet & Résultats 
    """ 

    st.header("Introduction")

    """
    Le Dataset de notre projet provient du site Kaggle https://www.kaggle.com/retailrocket/ecommerce-dataset, dans ce jeu de données nous avons les données comportementales des utilisateurs : **visitorid**, **events**, **timestamps**, **transactionid** et **itemid**.     
    L’objectif de ce projet est d’analyser le **comportement** des **acheteurs** d’un site e-commerce.
    Comprendre les habitudes de consommation de nos acheteurs afin d’observer des **similitudes** et des **disparités** entre les différents **groupes**.
    Notre problématique est que nous avons peu de variables et avons du **explorer** la variable **timestamp**.
    L'enjeu est d'analyser les différents groupe d’acheteurs via un modèle de **clustering**. 
    """ 

    # DATASET DF_SAMPLE
    df_sample = pd.read_csv('csv/df_sample.csv')
    df_sample = df_sample.fillna(0)
    st.write("Voici un aperçu du dataset: ",  "Nous avons pris le soin de remplacer les **NaN** par des **zéros** afin de ne perdre aucune information")
    st.write(df_sample)

    """
    # Premières analyses & Dataviz 
    Regardons la répartition de la variable "event" de notre Dataset.
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
    st.write("Ce premier graphique permet de comprendre que l’essentiel de notre dataset est constitué de **vues soit 97%** et **2,3%** **d'ajouts au panier**. A contrario, l’évènement « **transaction** » ne représente même pas **1%**. Cette analyse nous pousse à nous intéresser au comportement des **visiteurs** ainsi que les **acheteurs**. (voir graph ci-dessous)")

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

    st.write("D'après ces deux histogrammes, nous avons remarqué que les visiteurs avaient effectué le plus d’actions au mois de **juillet** (quid promotion ?). Concernant les jours de visites c'est assez homogène sauf le dernier jour du mois (**30 et 31**)")

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

    st.write(" Nous observons un comportement majoritaire de la part des visiteurs : ils  prennent **moins d’une heure à effectuer un achat**."
    "On constate aussi qu’une grande partie de nos visiteurs effectuent leurs achats en **moins de 10 minutes**. Partant de ce constat nous avons décidé d'observer l'achat en moins de 10 minutes, les acheteurs achètent **entre 1 et 3 minutes environ**.")

    
    """
    # Temps total en fonction du nombre de visites
    """

    image= Image.open('images/nuage.jpg') 
    st.image(image, width=698)   
    st.write("Ce premier nuage de points permet d'identifier **trois groupes d'individus**, " 
             "la somme totale du temps passé sur le site en fonction du nombre de visites. (variables choisies de manière arbitraire) "

             
    """
    # Analyse des items achetés 
    """

    image= Image.open('images/top_itemid.jpg') 
    st.image(image, width=698)   
    st.write("Cet histogramme affiche les 15 items les plus achetés, ainsi que le nombre de fois où ils ont été achetés." 
             "Ainsi, on constate que 3 produits se démarquent fortement. Ces 3 items ont pour itemid **461686**, **119736**, **213834**"
      
    image= Image.open('images/top3.jpg') 
    st.image(image, width=698)   
    st.write("On constate une **croissance** des ventes pour **l’item 461686**, avec un **pic au mois d’août**, "
             "accompagné d’une **légère baisse** des ventes au **mois de septembre**. Il s‘agit peut-être d’un item dit **« saisonnier »** "
             "(ex : ventilateur). L’intérêt des visiteurs pour l’item 119736 a été assez régulier tout au long des mois, "
             "avec tout de même une **baisse durant le mois de septembre**." 
             "Enfin, on constate que **l’item 213834 n’a pas été acheté au mois de mai.**" 
             "Cependant, il a été acheté de nombreuses fois au **mois de juillet** puis **peu de fois en août**. "        
             
             

    elif menu == "Présentation des 3 dataset":

    """
    # Présentation des 3 dataset
    """

    dataset = st.radio(
        "Choisissez votre dataset",
        ('Transactions Time', 'Visiteurs', 'Items'))
        
    if dataset == 'Transactions Time':

        st.header('Vous avez sélectionné : Dataset Transactions-Time')
        st.write("Voici un aperçu du dataset Transactions")

        time_sum_tran_sample = pd.read_csv('csv/time_sum_tran_sample.csv')
        st.write(time_sum_tran_sample)

        """
        # Résultat des clusterings
        """

        st.subheader('Observons les résultats des clusterings en fonction du nombre de clusters !')
        nclustrers = st.slider('Choisissez votre nombre de clusters', 2, 9, 2)
        st.write("Vous avez choisi le nombre de clusters égal à : ", nclustrers)

        k1 = KMeans(n_clusters=50).fit(time_sum_tran_sample)
        ac = AgglomerativeClustering(n_clusters=nclustrers).fit(k1.cluster_centers_)
        cd = pd.DataFrame(k1.cluster_centers_)

        time_sum_tran_sample['kmean1_label'] = k1.labels_

        for i in list(cd.index):
            time_sum_tran_sample.loc[time_sum_tran_sample['kmean1_label'] == cd.index[i], 'agglo_label'] = ac.labels_[i]

        new_centroids = time_sum_tran_sample.groupby('agglo_label').mean()
        new_time_sum_tran_sample = time_sum_tran_sample.drop(['agglo_label'], axis=1)

        k2 = KMeans(n_clusters=nclustrers, init=new_centroids)
        k2.fit(new_time_sum_tran_sample)

        k2_centroids = k2.cluster_centers_
        k2_labels = k2.labels_

        time_sum_tran_sample['kmean2_label'] = k2.labels_

        fig, ax = plt.subplots()
        sns.scatterplot(data=time_sum_tran_sample, x="sum_time_hour", y=time_sum_tran_sample.index, hue="kmean2_label")
        st.pyplot(fig)
             
        st.write("Dans le cluster 0, les consommateurs ont tendance à passer entre 15h et 23h pour effectuer une transaction,"
                 "pour le cluster 1 nous sommes entre 5h et 14h, enfin le cluster 2 la tendance se situe entre moins d’une heure et 4h.")

        # DENDROGRAMME
        st.subheader("Vérifions le nombre de cluster optimal (n=3) en utilisant un dendrogramme comme celui-ci :")  
        fig, ax = plt.subplots()
        Z = linkage(k1.cluster_centers_, method='ward', metric='euclidean')
        dendrogram(Z, leaf_rotation=70.)
        st.pyplot(fig)

    elif dataset == 'Visiteurs':

        st.header('Vous avez sélectionné : Dataset Visiteurs')
        st.write("Voici un aperçu du dataset Visiteurs")

        stats_sample = pd.read_csv('csv/stats_sample.csv')
        st.write(stats_sample)

        """
        # Résultat des clusterings
        """

        st.subheader('Observons les résultats des clusterings en fonction du nombre de clusters !')
        nclustrers = st.slider('Choisissez votre nombre de clusters', 2, 9, 2)
        st.write("Vous avez choisi le nombre de clusters égal à : ", nclustrers)

        k1 = KMeans(n_clusters=50).fit(stats_sample)
        ac = AgglomerativeClustering(n_clusters=nclustrers).fit(k1.cluster_centers_)
        cd = pd.DataFrame(k1.cluster_centers_)

        stats_sample['kmean1_label'] = k1.labels_

        for i in list(cd.index):
            stats_sample.loc[stats_sample['kmean1_label'] == cd.index[i], 'agglo_label'] = ac.labels_[i]

        new_centroids = stats_sample.groupby('agglo_label').mean()
        new_stats_sample = stats_sample.drop(['agglo_label'], axis=1)

        k2 = KMeans(n_clusters=nclustrers, init=new_centroids)
        k2.fit(new_stats_sample)

        k2_centroids = k2.cluster_centers_
        k2_labels = k2.labels_

        stats_sample['kmean2_label'] = k2.labels_

        fig, ax = plt.subplots()
        sns.scatterplot(data=stats_sample, x="sum_time_hour", y="nb_transactions", hue="kmean2_label")
        st.pyplot(fig)
             
        st.write(" Les visiteurs du cluster 2 ont en grande majorité passé entre 0 et 500 heures environ à effectuer leurs achats."
                 "Ceux du cluster 1 ont passé entre 500 et 1500 heures, tandis que ceux du cluster 0 ont passé entre 1500 et 3000 heures.")
                 


        # DENDROGRAMME
        st.subheader("Vérifions le nombre de cluster optimal (n=3) en utilisant un dendrogramme comme celui-ci :")  
        fig, ax = plt.subplots()
        Z = linkage(k1.cluster_centers_, method='ward', metric='euclidean')
        dendrogram(Z, leaf_rotation=70.)
        st.pyplot(fig)
             

    else:
        st.header('Vous avez sélectionné : Dataset Items')
        st.write("Voici un aperçu du dataset Items")
        top_produits_merged_buy_sample = pd.read_csv('csv/top_produits_merged_buy_sample.csv')
        st.write(top_produits_merged_buy_sample)

        """
        # Résultat des clusterings
        """
        st.subheader('Observons les scores et résultats des clusterings en fonction du nombre de clusters !')
        ### SCORING SILHOUETTE AVEC LE TABLEAU top_produits_merged_buy ####
        nclustrers = st.slider('Choisissez votre nombre de clusters', 2, 9, 2)
        st.write("Vous avez choisi le nombre de clusters égal à : ", nclustrers)

        with st.spinner('Caclul du coefficient de silhouette en cours...'):
            time.sleep(2)
            cluster = AgglomerativeClustering(n_clusters = nclustrers)
            cluster.fit(top_produits_merged_buy_sample)
            labels = cluster.labels_
            s_score = silhouette_score(top_produits_merged_buy_sample, labels, metric='sqeuclidean')
        st.write("**Score obtenu** : ",s_score)

        cluster = AgglomerativeClustering(n_clusters = nclustrers)
        cluster.fit(top_produits_merged_buy_sample)
        labels = cluster.labels_

        top_produits_merged_buy_sample['label_clustering'] = labels

        fig, ax = plt.subplots()
        sns.scatterplot(data=top_produits_merged_buy_sample, y="nb_transactions", x=top_produits_merged_buy_sample.index, hue="label_clustering",)
        st.pyplot(fig)
             
        image= Image.open('images/resume.jpg') 
        st.image(image, width=698)       
             
else:
    st.write("")



