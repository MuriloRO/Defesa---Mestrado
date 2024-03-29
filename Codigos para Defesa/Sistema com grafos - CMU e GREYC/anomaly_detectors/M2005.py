import numpy as np
import pandas as pd
from anomaly_detectors.base import UserModel, ClassificationAlgorithm
from adaptive_methods import adaptive_strategies
import ipdb

class M2005(ClassificationAlgorithm):
    def __init__(self, parameters = dict(), name='M2005'):
        self.name = name
        self.parameters = parameters

    def train(self, training_data):
        user_model = UserModel_(name='M2005', features=training_data)
        model_params = dict()
        
        #import pdb 
        #pdb.set_trace()
        
        for feature in training_data:
            
            #import pdb 
            #pdb.set_trace()
            
            if feature == 'ROUBO':
                model_params[feature] = training_data.iloc[1, -1]
                
            else:
                if training_data[feature].mean() == 0:

                    lower = min(training_data[feature].mean(), training_data[feature].median()) * (0.95 - (training_data[feature].std() / 0.00001))
                    upper = max(training_data[feature].mean(), training_data[feature].median()) * (1.05 + (training_data[feature].std() / 0.00001))


                else:
                    lower = min(training_data[feature].mean(), training_data[feature].median()) * (0.95 - (training_data[feature].std() / training_data[feature].mean()))
                    upper = max(training_data[feature].mean(), training_data[feature].median()) * (1.05 + (training_data[feature].std() / training_data[feature].mean()))


                model_params[feature] = (lower, upper)

        user_model.update(model=model_params)
        
        #import pdb;pdb.set_trace()

            
        return user_model


    def test(self, sample, user_model, decision_threshold, genuine_user, adaptive_TESTE):
        
        #import pdb 
        #pdb.set_trace()
        
        score = self.score(sample=sample, user_model=user_model, genuine_user=genuine_user, adaptive_TESTE=adaptive_TESTE )
        #print(f'score:{score}, decision:{decision_threshold}')
        
        if score > decision_threshold:
            return 1
        else:
            return 0



    def score(self, sample, user_model, genuine_user,adaptive_TESTE):
        
        #import pdb 
        #pdb.set_trace()
        
        match_sum = 0
        previousDimMatched = False
        
        #import ipdb;ipdb.set_trace()
        
        if adaptive_TESTE != 'False':
            if "ROUBO" in sample.index:
                if sample["ROUBO"] == genuine_user:
                    return 1
        
            for dim in user_model.keys():
                #import pdb; pdb.set_trace()
                if dim == "ROUBO":
                    continue

                if (sample[dim] <= user_model[dim][1]) and (sample[dim] >= user_model[dim][0]):
                    if previousDimMatched:
                        match_sum = match_sum + 1.5
                    else:
                        match_sum = match_sum + 1.0
                    previousDimMatched = True
                else:
                    previousDimMatched = False
        else:
            for dim in user_model.keys():

                if dim == "ROUBO":
                    continue
                #import pdb; pdb.set_trace()
                if (sample[dim] <= user_model[dim][1]) and (sample[dim] >= user_model[dim][0]):
                    if previousDimMatched:
                        match_sum = match_sum + 1.5
                    else:
                        match_sum = match_sum + 1.0
                    previousDimMatched = True
                else:
                    previousDimMatched = False
        
        #import pdb 
        #pdb.set_trace()
        
        max_sum = 1.0 + 1.5 * (len(user_model.keys()) -1)
        score = match_sum/max_sum

        return score 

class UserModel_(UserModel):
    def update(self, model):
        self.model = model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
