import pandas as pd
import numpy as np

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

        #import pdb;pdb.set_trace()

        return dic_referencias_impostoras         


                    
    def enrollment_grafos(self, dataset, adaptive):

        df = copy.deepcopy(dataset)

        self.users = dict()
        print("Cadastrando amostras biométricas no sistema...")
        
        #import pdb;pdb.set_trace()
            
        for i, user in enumerate(df.keys()):
            
            #import pdb;pdb.set_trace()
            
            train_size = df[user].shape[0]

            if adaptive == "GrowingWindow":
                self.users[user] = {
                    'biometric_reference' : self.detector.train(training_data=df[user].iloc[:train_size]),
                    'model' : self.detector,
                    'adaptive' : adaptive
                    
                }
                    #'impostor_reference': self.impostor_references(self.detector, df, user,train_size)
                
            elif adaptive == "GraphMinCutSliding":
                self.users[user] = {
                    'biometric_reference' : self.detector.train(training_data=df[user].iloc[:train_size]),
                    'model' : self.detector,
                    'adaptive' : adaptive
            }
       # import pdb;pdb.set_trace()
        return self

                    
      
    def autenticate_grafos(self, genuine_user, test_stream, decision_threshold, adaptive_TESTE, return_scores=False):
        
        decision = np.mean([*decision_threshold.values()])
        
        quantidade_do_batch = 25
        # Faço essa cópia para não modificar a referência biométrica em testes de fluxos de dados de outros cenários.
        
        biometric_reference = copy.deepcopy(self.users[genuine_user]['biometric_reference'])
        
        
        lista_nao_usadas = list() #teste
        lista_usadas = list() #teste
        
        
        y_pred = list()
        stream_scores = list()
        pred_gen = list()
        pred_imp = list()
        matriz_de_adj_cosseno = list()
        
        count = 0

        # Iniciando avaliacao do fluxo de dados de teste
        for _, features in test_stream.iterrows():
            

            lista_usadas.append(features)
            pred_gen.append(features.values.tolist())
                
            count+=1
            
            if count == quantidade_do_batch:
                
                galeria_user = dict()
                galeria_user = self.users[genuine_user]['biometric_reference'].features.values.tolist()
                
                #import pdb; pdb.set_trace()
                #adicionar o tamanho da galeria
                lista_de_amostras = self.Lista_de_amostras(pred_gen, galeria_user)
                
                #import pdb; pdb.set_trace()   
                matriz_de_adj_cosseno = self.Matriz_de_adj_cosseno(lista_de_amostras)
                
                #len(pred_gen) teria que ser o tamanho da galeria genuina
                ponto1_cosseno = self.Ponto_mais_distante_cosseno(len(galeria_user), lista_de_amostras)
                
                #len(pred_gen) teria que ser o tamanho da galeria genuina
                _,partition_mc_cosseno = self.Gera_grafo_cosseno(len(galeria_user), ponto1_cosseno, matriz_de_adj_cosseno)
                
                genuinos_adaptacao = self.Particao_genuina(partition_mc_cosseno,lista_de_amostras)
                
                for i in range(len(pred_gen)):    
                    if pred_gen[i] in genuinos_adaptacao:
                        
                        y_pred.append(1)

                        biometric_reference = adaptive_strategies.update(strategy=self.users[genuine_user]['adaptive'],
                                                                            detector=self.detector,
                                                                            biometric_reference=biometric_reference, 
                                                                            new_features=pred_gen[i]) 
                        
                        
                    else:
                        y_pred.append(0)


                count=0
                pred_gen = list()

        return y_pred , lista_nao_usadas, lista_usadas
                

      

 #Funcoes de auxilio para o graph min cut        

    def Particao_genuina(self,partition_mc_cosseno,lista_de_amostras):
        
        #particao_impostora = list(partition_mc_euclidiano[1])
        #particao_impostora.remove('impo')
        
        particao_genuina = list(partition_mc_cosseno[0])
        particao_genuina.remove('genu')

        genuinos_adaptacao = [lista_de_amostras[i] for i in particao_genuina ]
        
        return genuinos_adaptacao
    
    
    def Lista_de_amostras(self, pred_gen,galeria_genuina):
        Array_de_amostras = list()
               
        #import pdb;pdb.set_trace()
        
        for i in range(len(pred_gen)):
            Array_de_amostras.append(pred_gen[i][0:])
        
        for k in range(len(galeria_genuina)):
            Array_de_amostras.append(galeria_genuina[k][0:])

        
        
        return Array_de_amostras


    def Ponto_mais_distante_cosseno(self,quantidade_de_genuinos, Array_de_amostras):

        Y = Array_de_amostras[quantidade_de_genuinos:]
        X = Array_de_amostras[0:quantidade_de_genuinos]

        ponto1_cosseno, ponto2_cosseno = self.Distancia_cosseno(Y,X)
        ponto1_cosseno = ponto1_cosseno + quantidade_de_genuinos

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

        #import pdb;pdb.set_trace()

        return dic_referencias_impostoras       
