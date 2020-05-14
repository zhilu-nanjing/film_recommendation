import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math, nltk, warnings
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from wordcloud import WordCloud, STOPWORDS

gaussian_filter = lambda x,y,sigma: math.exp(-(x-y)**2/(2*sigma**2))


def entry_variables(df, id_entry):
    """
    将director_name、actor_NUM_name、plot_keywords抽取出来放到一个list里
    :param df:
    :param id_entry: <class 'int'> 12
    :return: col_labels:<class 'list'>: ['Christopher Nolan', 'Michael Caine', 'Gary Oldman', 'Anne Hathaway', 'dc comics', 'crime fighter', 'terrorist', 'secret identity', 'burglar', 'hostage drama', 'gotham city', 'vigilante', 'cover-up', 'superhero', 'tragic hero', 'terror', 'death', 'imax', 'flood']
    """
    col_labels = []
    if pd.notnull(df['director_name'].iloc[id_entry]):
        for s in df['director_name'].iloc[id_entry].split('|'):
            col_labels.append(s)

    for i in range(3):
        column = 'actor_NUM_name'.replace('NUM', str(i + 1))
        if pd.notnull(df[column].iloc[id_entry]):
            for s in df[column].iloc[id_entry].split('|'):
                col_labels.append(s)

    if pd.notnull(df['plot_keywords'].iloc[id_entry]):
        for s in df['plot_keywords'].iloc[id_entry].split('|'):
            col_labels.append(s)
    return col_labels

def add_variables(df, REF_VAR):
    for s in REF_VAR: df[s] = pd.Series([0 for _ in range(len(df))])
    colonnes = ['genres', 'actor_1_name', 'actor_2_name',
                'actor_3_name', 'director_name', 'plot_keywords']
    for categorie in colonnes:
        for index, row in df.iterrows():
            if pd.isnull(row[categorie]): continue
            for s in row[categorie].split('|'):
                if s in REF_VAR: df.set_value(index, s, 1)
    return df

def recommand(df, id_entry):
    df_copy = df.copy(deep = True)
    liste_genres = set()
    for s in df['genres'].str.split('|').values:
        liste_genres = liste_genres.union(set(s))
    #_____________________________________________________
    # Create additional variables to check the similarity
    variables = entry_variables(df_copy, id_entry)
    variables += list(liste_genres)
    df_new = add_variables(df_copy, variables)
    #____________________________________________________________________________________
    # determination of the closest neighbors: the distance is calculated / new variables
    X = df_new.as_matrix(variables)
    nbrs = NearestNeighbors(n_neighbors=31, algorithm='auto', metric='euclidean').fit(X)

    distances, indices = nbrs.kneighbors(X)
    xtest = df_new.iloc[id_entry].as_matrix(variables)
    xtest = xtest.reshape(1, -1)

    distances, indices = nbrs.kneighbors(xtest)

    return indices[0][:]


def extract_parameters(df, liste_films):
    parametres_films = ['_' for _ in range(31)]
    i = 0
    max_users = -1
    for index in liste_films:
        parametres_films[i] = list(df.iloc[index][['movie_title', 'title_year',
                                                   'vote_average', 'num_user_for_reviews',
                                                   'num_voted_users']])
        parametres_films[i].append(index)
        max_users = max(max_users, parametres_films[i][4])
        i += 1

    title_main = parametres_films[0][0]
    annee_ref = parametres_films[0][1]
    parametres_films.sort(key=lambda x: critere_selection(title_main, max_users,
                                                          annee_ref, x[0], x[1], x[2], x[4]), reverse=True)

    return parametres_films

def sequel(titre_1, titre_2):
    if fuzz.ratio(titre_1, titre_2) > 50 or fuzz.token_set_ratio(titre_1, titre_2) > 50:
        return True
    else:
        return False


def critere_selection(title_main, max_users, annee_ref, titre, annee, imdb_score, votes):
    if pd.notnull(annee_ref):
        facteur_1 = gaussian_filter(annee_ref, annee, 20)
    else:
        facteur_1 = 1

    sigma = max_users * 1.0

    if pd.notnull(votes):
        facteur_2 = gaussian_filter(votes, max_users, sigma)
    else:
        facteur_2 = 0

    if sequel(title_main, titre):
        note = 0
    else:
        note = imdb_score ** 2 * facteur_1 * facteur_2

    return note

def add_to_selection(film_selection, parametres_films):
    film_list = film_selection[:]
    icount = len(film_list)
    for i in range(31):
        already_in_list = False
        for s in film_selection:
            if s[0] == parametres_films[i][0]: already_in_list = True
            if sequel(parametres_films[i][0], s[0]): already_in_list = True
        if already_in_list: continue
        icount += 1
        if icount <= 5:
            film_list.append(parametres_films[i])
    return film_list

def remove_sequels(film_selection):
    removed_from_selection = []
    for i, film_1 in enumerate(film_selection):
        for j, film_2 in enumerate(film_selection):
            if j <= i: continue
            if sequel(film_1[0], film_2[0]):
                last_film = film_2[0] if film_1[1] < film_2[1] else film_1[0]
                removed_from_selection.append(last_film)

    film_list = [film for film in film_selection if film[0] not in removed_from_selection]

    return film_list

def find_similarities(df, id_entry, del_sequels = True, verbose = False):
    if verbose:
        print(90*'_' + '\n' + "QUERY: films similar to id={} -> '{}'".format(id_entry,
                                df.iloc[id_entry]['movie_title']))
    #____________________________________
    liste_films = recommand(df, id_entry)
    #__________________________________
    # Create a list of 31 films
    parametres_films = extract_parameters(df, liste_films)
    #_______________________________________
    # Select 5 films from this list
    film_selection = []
    film_selection = add_to_selection(film_selection, parametres_films)
    #__________________________________
    # delation of the sequels
    if del_sequels: film_selection = remove_sequels(film_selection)
    #______________________________________________
    # add new films to complete the list
    film_selection = add_to_selection(film_selection, parametres_films)
    #_____________________________________________
    selection_titres = []
    for i,s in enumerate(film_selection):
        selection_titres.append([s[0].replace(u'\xa0', u''), s[5]])
        if verbose: print("nº{:<2}     -> {:<30}".format(i+1, s[0]))

    return selection_titres

import pickle
f = open('.\middle_data\df.pkl','rb')
df = pickle.load(f)
# dum = find_similarities(df, 12, del_sequels = False, verbose = True)
# print(dum)
# dum = find_similarities(df, 12, del_sequels = True, verbose = True)
# print(dum)
# dum = find_similarities(df, 2, del_sequels = True, verbose = True)
# print(dum)
selection = dict()
for i in range(0, 20, 3):
    selection[i] = find_similarities(df, i, del_sequels = True, verbose = True)