from biometric_system_cosseno import BiometricSystem
from anomaly_detectors.M2005 import M2005 
from anomaly_detectors import thresholds
from data_stream import data_stream
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import KFold

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import networkx as nx


import json
import ipdb
import os, sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import random
import copy
import IPython.display as ipd


def Average(lst):
    x = sum(lst)
    y = len(lst)
    a = (round(x,6) / round(y))
    return a


def split_data_enrollment(dataset, column, n_samples):
    data_to_enrollment = dict()

    for value in dataset[column].unique():
        
        data_to_enrollment.setdefault(value, dataset.loc[dataset[column]==value].iloc[:(n_samples//2)].loc[:,~dataset.columns.isin([column])].reset_index(drop=True))
    
    return data_to_enrollment

def split_data_validation(dataset, column, n_samples):
    data_to_validation = dict()

    for value in dataset[column].unique():
        
        data_to_validation.setdefault(value, dataset.loc[dataset[column]==value].iloc[(n_samples//2):n_samples].loc[:,~dataset.columns.isin([column])].reset_index(drop=True))
    
    return data_to_validation


def split_data_recognition(dataset, column, n_samples):
    data_to_recognition = dict()

    for value in dataset[column].unique():
 
        data_to_recognition.setdefault(value, dataset.loc[dataset[column]==value].iloc[:n_samples].loc[:,~dataset.columns.isin([column])].reset_index(drop=True))
    
    
    return data_to_recognition

def split_data(dataset, column, n_samples):
    data_to_enrollment = dict()
    data_to_validation = dict()
    data_to_recognition = dict()

    for value in dataset[column].unique():
        
        #Treinamento, pegando dados de usuarios 
        data_to_enrollment.setdefault(value, dataset.loc[dataset[column]==value].iloc[:(n_samples//2)].loc[:,~dataset.columns.isin([column])].reset_index(drop=True))

        # Definir o limiar de decisao dos algoritmos de classificacao
        data_to_validation.setdefault(value, dataset.loc[dataset[column]==value].iloc[(n_samples//2):n_samples].loc[:,~dataset.columns.isin([column])].reset_index(drop=True))

        # Criar o fluxo de teste
        data_to_recognition.setdefault(value, dataset.loc[dataset[column]==value].iloc[n_samples:].loc[:,~dataset.columns.isin([column])].reset_index(drop=True))
    
    
    return data_to_enrollment,data_to_validation, data_to_recognition


if __name__ == '__main__':
    print("INICIO")

    df = pd.read_csv('dados/DSL-Modificado.csv', delimiter = ',', index_col = [0])
    df = df.drop(['rep'], axis=1)
    users = df['subject'].unique()

    perc = 0.5
    impostor_rate = 0.30
    rate_external_impostor = 0
    R=1
    GRAPH_MIN_CUT_SLIDING = []

    #--------------------------------------------------------------------------------------------------------------# 
    #Separação de index, primeiro pra treino, segundo para validação e teste
    # Dependendo do teste é necessario trocar a linha ("Separação por Index")

    sessionIndex1 = 1
    sessionIndex2 = 2

    #--------------------------------------------------------------------------------------------------------------# 
    #Separação de usuarios

    len_reg_users = int(len(users) * perc)

    kfold = KFold(n_splits=2, shuffle=True, random_state=R)
    splits = kfold.split(users)

    #--------------------------------------------------------------------------------------------------------------# 
    #Registro de usuarios

    for i, (reg_users, not_reg_users) in enumerate(splits):

        internal_users = copy.deepcopy(df.loc[df['subject'].isin(users[reg_users])])
        external_users = copy.deepcopy(df.loc[~df['subject'].isin(users[reg_users])])

    #--------------------------------------------------------------------------------------------------------------#  
    #Separação por Index

    dataS1 = internal_users.loc[(internal_users['sessionIndex'] == sessionIndex1)]
    dataS1.drop(["sessionIndex"], axis=1, inplace=True)

    dataS2 = internal_users.loc[(internal_users['sessionIndex'] != sessionIndex1)]
    dataS2.drop(["sessionIndex"], axis=1, inplace=True)

    #--------------------------------------------------------------------------------------------------------------#  
    # Dados para Treino, Validação e Reconhecimento

    #Treinamento, pegando dados de usuarios 
    data_to_enrollment = split_data_enrollment(dataS1, column='subject', n_samples=50)

    # Definir o limiar de decisao dos algoritmos de classificacao
    data_to_validation = split_data_validation(dataS1, column='subject', n_samples=50)

    # Criar o fluxo de teste
    data_to_recognition = split_data_recognition(dataS2, column='subject', n_samples=350)

    _, _, external_users_data = split_data(external_users, column='subject', n_samples=50)



    #Sistema com Adaptação (GraphMinCut)

    detector = M2005()
    adaptive= "GraphMinCutSliding"
    system = BiometricSystem(detector=detector, random_state=R)
    system.enrollment_grafos(dataset=data_to_enrollment, adaptive=adaptive)

    decision_threshold = thresholds.best_threshold(data_to_validation, system, size=10, random_state=R)

    metrics_adaptativo_grafo_sliding = dict()

    lista_nao_usadas_grafo_sliding = list()
    lista_usadas_grafo_sliding = list()



    for j, genuine in enumerate(system.users.keys()):


        ipd.clear_output(wait=True)
        print(f"Rodando GraphMinCutSliding")
        print(f"Testando usuário {j+1}/{len(system.users.keys())}")

        datastream = data_stream.Random(impostor_rate= impostor_rate,
                                        rate_external_impostor=rate_external_impostor,
                                        random_state=R)

        test_stream, y_true, amostras_grafo_sliding_genuinas,amostras_grafo_sliding_impostoras = datastream.create(genuine,
                                                                                                    data_to_recognition,
                                                                                                    external_users_data)



        y_pred, lista_nao_usadas_grafo_sliding2, lista_usadas_grafo_sliding2,batch = system.autenticate_grafos(genuine,
                                                                                    test_stream,
                                                                                    decision_threshold=decision_threshold,
                                                                                    adaptive_TESTE=adaptive,
                                                                                    return_scores=False)

        lista_nao_usadas_grafo_sliding.append(lista_nao_usadas_grafo_sliding2)
        lista_usadas_grafo_sliding.append(lista_usadas_grafo_sliding2)

        fmr, fnmr, b_acc = system.compute_metrics(y_true, y_pred)

        for met in ['fmr','fnmr','b_acc']:
            metrics_adaptativo_grafo_sliding.setdefault(genuine, dict()).setdefault(met,[]).append(eval(met))
        json.dump(metrics_adaptativo_grafo_sliding, open("metricas_grafo_sliding.json", "w"))


    usuarios = metrics_adaptativo_grafo_sliding.keys()
    result = pd.DataFrame(metrics_adaptativo_grafo_sliding.values())

    fmr_mean = []
    fnmr_mean = []
    b_acc_mean = []

    for i in result['fmr']:
        fmr_mean.append(Average(i))

    for i in result['fnmr']:
        fnmr_mean.append(Average(i))

    for i in result['b_acc']:
        b_acc_mean.append(Average(i))

    metrics_adaptativo_grafo_sliding_mean = pd.DataFrame(list(zip(usuarios, fmr_mean, fnmr_mean,b_acc_mean)),
                columns =['Usuarios','fmr_mean', 'fnmr_mean','b_acc_mean'])

    GRAPH_MIN_CUT_SLIDING.append(metrics_adaptativo_grafo_sliding_mean['b_acc_mean'].mean())

    print('---------------------')
    print("Teste com sessao:", sessionIndex1)
    print("Teste com batch:", batch)
    print("Acuracia do GRAPH_MIN_CUT_SLIDING:", GRAPH_MIN_CUT_SLIDING)