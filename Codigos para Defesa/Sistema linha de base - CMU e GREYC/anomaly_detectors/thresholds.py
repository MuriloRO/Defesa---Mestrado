import numpy as np
import pandas as pd
import IPython.display as ipd

def best_threshold(dataset, system, size, random_state):
    
    thresholds = dict()
    #import ipdb;ipdb.set_trace()

    for i, genuine in enumerate(system.users.keys()):
        ipd.clear_output(wait=True)
        print("Configurando limiares de decisão:")
        print(f"Usuário: {i+1}/{len(dataset.keys())} ")
        
        impostor_df = list()
        for user in dataset.keys():
            impostor_df.append(dataset[user])
            
        #import ipdb;ipdb.set_trace()
        
        impostor_df = pd.concat(impostor_df,ignore_index=True)
        impostor_data  = impostor_df.sample(size, replace=False, random_state=random_state).sort_index() 
        
        #import ipdb;ipdb.set_trace()
        
        model = system.users[genuine]['model']
        
        adaptive_TESTE = system.users[genuine]['adaptive']
        
        #import ipdb;ipdb.set_trace()
        
        reference = system.users[genuine]['biometric_reference'][0] if type(system.users[genuine]['biometric_reference']) == tuple else system.users[genuine]['biometric_reference']
        if model.name == 'M2005':
            score_type = 'similarity'
            genuine_scores = dataset[genuine].apply(lambda x: model.score(x, reference.model, genuine, adaptive_TESTE), axis=1) 
            impostor_scores = impostor_data.apply(lambda x: model.score(x, reference.model, genuine, adaptive_TESTE), axis=1)
        else:
            Exception("Choose a valid model.")
        
        #import pdb 
        #pdb.set_trace()
        
        decision_threshold = calculate_best_threshold(user_scores=genuine_scores, impostor_scores=impostor_scores, score_type=score_type)
        
        
        thresholds[genuine] = decision_threshold
    
    return thresholds

def calculate_best_threshold(user_scores, impostor_scores, score_type):
    
    predictions = list(user_scores) + list(impostor_scores)
    labels = np.concatenate((np.ones(len(user_scores)), np.zeros(len(impostor_scores))))
    best_bacc = -float("inf")
    for score in predictions:
        _, _, bacc = reporter(scores=predictions, true_labels=labels, threshold=score, score_type=score_type)
        if bacc > best_bacc:
            best_bacc = bacc
            decision = score
            
        #import ipdb
        #ipdb.set_trace()
    
    return decision

def reporter(scores, true_labels, threshold, score_type):
    y_genuine = list()
    y_impostor = list()

    if score_type=='distance':
        y_genuine = [1 if scores[i] < threshold else 0 for i, sample in enumerate(true_labels) if sample == 1]
        y_impostor = [1 if scores[i] < threshold else 0 for i, sample in enumerate(true_labels) if sample == 0]

    elif score_type=='similarity':
        y_genuine = [1 if scores[i] > threshold else 0 for i, sample in enumerate(true_labels) if sample == 1]
        y_impostor = [1 if scores[i] > threshold else 0 for i, sample in enumerate(true_labels) if sample == 0]
    else:
        Exception("Choose a valid score_type.")

    fnmr= 1- sum(y_genuine) / len(y_genuine)
    fmr = sum(y_impostor) / len(y_impostor)
    bacc = 1- (fnmr + fmr)/2
    return fmr, fnmr, bacc
