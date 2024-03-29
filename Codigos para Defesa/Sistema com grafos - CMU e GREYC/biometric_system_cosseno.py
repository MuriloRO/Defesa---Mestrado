import pandas as pd
import numpy as np
from itertools import chain
from random import sample


from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import networkx as nx

from adaptive_methods import adaptive_strategies

import IPython.display as ipd
import copy

class BiometricSystem:

    def __init__(self, detector, random_state):
        self.detector = detector
        self.random_state = random_state

    def compute_metrics(self, y_true, y_pred):
        if type(y_pred) == list:
            y_pred = pd.Series(y_pred)
        y_genuine = y_pred[y_true==1]
        y_impostor = y_pred[y_true==0]

        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - ((FNMR + FMR) / 2.0)
        return FMR, FNMR, B_acc, y_genuine , y_impostor

    def compute_metrics_scores(self, y_true, y_scores, decision_threshold):
        y_genuine = [1 if y_scores[i] > decision_threshold else 0 for i, target in enumerate(y_true) if target == 1]
        y_impostor = [1 if y_scores[i] > decision_threshold else 0 for i, target in enumerate(y_true) if target == 0]

        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - ((FNMR + FMR) / 2.0)
        return FMR, FNMR, B_acc

    def impostor_references(self, detector, df, user,train_size):

        #Implementa a galeria negativa

        df_momentaneo = copy.deepcopy(df)
        df_momentaneo.pop(user)     

        dic_referencias_impostoras = dict()

        for i, usuarios in enumerate(df_momentaneo.keys()):

            dic_referencias_impostoras[usuarios] = self.detector.train(training_data=df[usuarios].iloc[:train_size])

        return dic_referencias_impostoras         


                    
    def enrollment_grafos(self, dataset, adaptive):

        df = copy.deepcopy(dataset)

        self.users = dict()
        print("Cadastrando amostras biométricas no sistema...")
        
        #import pdb;pdb.set_trace()
            
        for i, user in enumerate(df.keys()):
            
            #verificar esse train size
            train_size = df[user].shape[0]
            
            if adaptive == "GrowingWindow":
                self.users[user] = {
                    'biometric_reference' : self.detector.train(training_data=df[user].iloc[:train_size]),
                    'model' : self.detector,
                    'adaptive' : adaptive,
                    'impostor_reference': self.impostor_references(self.detector, df, user,train_size)
                }     
                
            elif adaptive == "SlidingWindow":
                self.users[user] = {
                    'biometric_reference' : self.detector.train(training_data=df[user].iloc[:train_size]),
                    'model' : self.detector,
                    'adaptive' : adaptive,
                    'impostor_reference': self.impostor_references(self.detector, df, user,train_size)
                }

                
        return self

                    
      
    def autenticate_grafos(self, genuine_user, test_stream, decision_threshold, adaptive_TESTE, return_scores=False):

        lista_nao_usadas = list() #teste
        lista_usadas = list() #teste
        
        
        y_pred = list()
        stream_scores = list()
        matriz_de_adj_cosseno = list()
        galeria_user = dict()

        decision = np.mean([*decision_threshold.values()])
        
        biometric_reference = copy.deepcopy(self.users[genuine_user]['biometric_reference'])
        
        
        # Iniciando avaliacao do fluxo de dados de teste
        for _, features in test_stream.iterrows():
            
            score = self.detector.score(sample=features, user_model=biometric_reference.model, genuine_user=genuine_user, adaptive_TESTE=adaptive_TESTE) 
            
            if score >= decision:

                lista_usadas.append(features)
                y_pred.append(1)
                

            else:
                lista_nao_usadas.append(features)
                y_pred.append(0)

            pred = features.values.tolist()
                            
            # Pega os dados de treinamento de usuarios que não são pertencentes a amostras analisada
        
            modelo_extra = copy.deepcopy(self.users[genuine_user]['impostor_reference'])     
            modelo_usermodel = list(modelo_extra.values())
            
            #para somente 25 amostras
            #x = modelo_usermodel.features.values.tolist()
            #amostra_extra = x

            #Para mais de 25
            extras = list()
            for i in range(len(modelo_usermodel)) : 
                extras.append(modelo_usermodel[i].features.values.tolist())   
            lista_total = list(chain(*extras))
            
            amostra_extra = sample(lista_total, 25)
            
            #import pdb;pdb.set_trace()
                
            #Galeria do usuarios  
            
            galeria_user = biometric_reference.features.values.tolist()
            
            
            # COLOCAR AS AMOSTRAS QUE VAO FORMAR O GRAFO (GALERIA DO USUARIO + pred_imp ou pred_gen + OUTRAS AMOSTRAS(ainda nao sei quais serao))
            lista_de_amostras = self.Lista_de_amostras(pred, amostra_extra, galeria_user)
            
            matriz_de_adj_cosseno = self.Matriz_de_adj_cosseno(lista_de_amostras)

            #len(pred_gen) teria que ser o tamanho da galeria genuina (problema esta nesse tamanho da galeria genuina, quando ela aumenta muito da ruim)
            ponto1_cosseno = self.Ponto_mais_distante_cosseno(len(galeria_user), lista_de_amostras)
            
            #len(pred_gen) teria que ser o tamanho da galeria genuina
            _,partition_mc_cosseno = self.Gera_grafo_cosseno(len(galeria_user), ponto1_cosseno, matriz_de_adj_cosseno)

            genuinos_adaptacao = self.Particao_genuina(partition_mc_cosseno, lista_de_amostras)
                        
            #Comparar as amostras de dentro de genuinos_adaptação com a amostra que veio do features

            
            if pred in genuinos_adaptacao:
                biometric_reference = adaptive_strategies.update(strategy=self.users[genuine_user]['adaptive'],
                                                                    detector=self.detector,
                                                                    biometric_reference=biometric_reference, 
                                                                    new_features=features)


        return y_pred , lista_nao_usadas, lista_usadas
                

      

 #Funcoes de auxilio para o graph min cut        

    def Particao_genuina(self,partition_mc_cosseno,lista_de_amostras):
        
        #particao_impostora = list(partition_mc_euclidiano[1])
        #particao_impostora.remove('impo')
        
        particao_genuina = list(partition_mc_cosseno[0])
        particao_genuina.remove('genu')

        genuinos_adaptacao = [lista_de_amostras[i] for i in particao_genuina ]
        
        return genuinos_adaptacao
    
    
    def Lista_de_amostras(self, pred_gen, amostras_extras,galeria_genuina):
        Array_de_amostras = list()
        
        for k in range(len(galeria_genuina)):
            Array_de_amostras.append(galeria_genuina[k][0:])
         
        Array_de_amostras.append(pred_gen)
        
        for j in range(len(amostras_extras)):
            Array_de_amostras.append(amostras_extras[j][0:])

        return Array_de_amostras


    def Ponto_mais_distante_cosseno(self,quantidade_de_genuinos, Array_de_amostras):

        Y = Array_de_amostras[quantidade_de_genuinos:]
        X = Array_de_amostras[0:quantidade_de_genuinos]

        
        ponto1_cosseno, ponto2_cosseno = self.Distancia_cosseno(Y,X)
        ponto1_cosseno = ponto1_cosseno + quantidade_de_genuinos
        
        #import pdb;pdb.set_trace()
        return ponto1_cosseno
    

    #FUNCOES PARA GRAFO COM SIMILARIDADE COSSENO

    def Distancia_cosseno(self, Y, X):
        
        
        cosine_dist = cosine_similarity(Y, X)

        # Utilizo o np.min para encontrar o nó menos similar
        result = np.where(cosine_dist == np.min(cosine_dist))


        ponto1 = int(result[0])
        ponto2 = int(result[1])

        return ponto1, ponto2


    def Matriz_de_adj_cosseno(self, Array_de_amostras):
        
        #import pdb;pdb.set_trace()

        Matriz_de_adjacencia = kneighbors_graph(Array_de_amostras, 3, mode='distance',metric='cosine', include_self=False)

        return Matriz_de_adjacencia.toarray()


    def Gera_grafo_cosseno(self, quantidade_de_genuinos, ponto1_cosseno, matriz_de_adj_cosseno):
    # Conecta todo mundo com os 3 mais proximos 
        G = nx.Graph()
        for i in range(0,matriz_de_adj_cosseno.shape[0]):
            for j in range(0,matriz_de_adj_cosseno.shape[1]):
                        if(matriz_de_adj_cosseno[i,j]!=0):
                            G.add_edge(i,j, capacity=(matriz_de_adj_cosseno[i,j]))


        # Conecta os quantidade_de_genuinos a um genuino infinito
        for k in range(0,quantidade_de_genuinos,1):
            G.add_edge('genu',k, capacity=99999999)


        # Conecta o ponto nao rotulado mais distante dos genuinos ao impostor de peso infinito 
        G.add_edge(ponto1_cosseno,'impo', capacity=9999999)

        # Calcula o corte minimo

        cut_value_mc, partition_mc = nx.minimum_cut(G, "genu", "impo")

        return cut_value_mc, partition_mc

    
    def impostor_references(self, detector, df, user,train_size):

        #Implementa a galeria negativa

        df_momentaneo = copy.deepcopy(df)
        df_momentaneo.pop(user)     

        dic_referencias_impostoras = dict()

        for i, usuarios in enumerate(df_momentaneo.keys()):

            dic_referencias_impostoras[usuarios] = self.detector.train(training_data=df[usuarios].iloc[:train_size])

        return dic_referencias_impostoras       


    
     
