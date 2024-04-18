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

    def enrollment(self, dataset, adaptive):

        df = copy.deepcopy(dataset)

        self.users = dict()
        print("Cadastrando amostras biométricas no sistema..")
            
        
        for i, user in enumerate(df.keys()):
            
            train_size = df[user].shape[0]
            
            if adaptive == "DoubleParallel":
                self.users[user] = {
                    'biometric_reference' : (self.detector.train(training_data=df[user].iloc[:train_size]),
                                            self.detector.train(training_data=df[user].iloc[:train_size])),
                    'model' : self.detector,
                    'adaptive' : adaptive            
                }
            else:
                self.users[user] = {
                    'biometric_reference' : self.detector.train(training_data=df[user].iloc[:train_size]),
                    'model' : self.detector,
                    'adaptive' : adaptive                    
                }
        return self

    def autenticate(self, genuine_user, test_stream, decision_threshold, adaptive_TESTE, return_scores=False):
        
        decision = np.mean([*decision_threshold.values()])
        
        
        # Faço essa cópia para não modificar a referência biométrica em testes de fluxos de dados de outros cenários.
        
        biometric_reference = copy.deepcopy(self.users[genuine_user]['biometric_reference'])        
        
        lista_nao_usadas = list()
        lista_usadas = list() 
        lista_adaptacao = list()
      
        
        if self.users[genuine_user]['adaptive']=='False':
            
            adaptive_TESTE = 'False'
            
            if return_scores:
                stream_scores = test_stream.apply(lambda x: self.detector.score(x, biometric_reference.model), axis=1) 
            else:
                y_pred = test_stream.apply(lambda x: self.detector.test(x, biometric_reference.model, decision, genuine_user, adaptive_TESTE=adaptive_TESTE), axis=1) 
                
        else:
            
            y_pred = list()
            stream_scores = list()

           
            # Iniciando avaliacao do fluxo de dados de teste
            for _, features in test_stream.iterrows():
                
                #import ipdb; ipdb.set_trace()
                
                #Etapa de geraçao score, gerando score para amostra de consulta
                if self.users[genuine_user]['adaptive'] == 'DoubleParallel':
                    gw_score = self.detector.score(sample=features, user_model=biometric_reference[0].model, genuine_user= genuine_user, adaptive_TESTE=adaptive_TESTE)
                    sw_score = self.detector.score(sample=features, user_model=biometric_reference[1].model, genuine_user= genuine_user, adaptive_TESTE=adaptive_TESTE)
                    score = (gw_score + sw_score) / 2
                    
                
                else:
                    score = self.detector.score(sample=features, user_model=biometric_reference.model, genuine_user=genuine_user, adaptive_TESTE=adaptive_TESTE)

                stream_scores.append(score)
                #Final da etapa de geraçao de score
                
                #import ipdb; ipdb.set_trace()

                
                #Classificação                               
                if score >= decision:
                    y_pred.append(1)

                else:
                    y_pred.append(0)
              
                
                #Adaptação
                if score >= decision: # or features['ROUBO'] == genuine_user:
                    lista_usadas.append(features)
                    lista_adaptacao.append(1)

                    biometric_reference = adaptive_strategies.update(detector=self.detector,
                                                    strategy=self.users[genuine_user]['adaptive'], 
                                                    biometric_reference=biometric_reference, 
                                                    new_features=features)
                else:
                    lista_nao_usadas.append(features)
                    lista_adaptacao.append(0)

                #Classificando entre genuino e impostor
  
        if return_scores:
            
            return stream_scores, lista_nao_usadas, lista_usadas
        else:

            return y_pred , lista_nao_usadas, lista_usadas,lista_adaptacao

        
    def compute_metrics(self, y_true, y_pred):
        #import ipdb; ipdb.set_trace()
        
        if type(y_pred) == list:
            y_pred = pd.Series(y_pred)
            
        y_genuine = y_pred[y_true==1]
        y_impostor = y_pred[y_true==0]

        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - ((FNMR + FMR) / 2.0)
        
       # import ipdb; ipdb.set_trace()
        return FMR, FNMR, B_acc, y_genuine,y_impostor
        
    

    def compute_metrics_scores(self, y_true, y_scores, decision_threshold):
        y_genuine = [1 if y_scores[i] > decision_threshold else 0 for i, target in enumerate(y_true) if target == 1]
        y_impostor = [1 if y_scores[i] > decision_threshold else 0 for i, target in enumerate(y_true) if target == 0]

        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - ((FNMR + FMR) / 2.0)
        return FMR, FNMR, B_acc

    def contagem_adaptacao(self, y_true, lista_adaptacao):
        if type(lista_adaptacao) == list:
            lista_adaptacao = pd.Series(lista_adaptacao)    
        
        adapt_genuine = lista_adaptacao[y_true==1]  
        adapt_impostor = lista_adaptacao[y_true==0]
        #import ipdb; ipdb.set_trace()    
        
        return adapt_genuine, adapt_impostor


           
        
        
        
                    


        
        






        
   
