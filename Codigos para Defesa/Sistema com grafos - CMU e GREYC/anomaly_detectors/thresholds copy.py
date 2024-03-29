import numpy as np
import pandas as pd

class FindThreshold:
    ''' Essa classe é responsável por encontrar o melhor limiar de decisão que separa amostras impostoras e genuínas para cada um dos
    classificadores utilizados.

    '''
    #esse random state faz diferença no adaptativo
    def __init__(self, random_state):
        #self.train = eval(detector+'detector')(name=detector, normalize=normalize, adaptive=False)
        #self.model = detector
        self.random_state = random_state

    def best_threshold(self, biometric_reference, detector, valid_data, valid_size, extern_impostors=list()):
        """ Calculus of best decision threshold for each user in `biometric_reference.keys()` and `detector`. Don't use external impostor data here."""
        thresholds = dict()

        # Um usuario é definido como genuino, enquanto os outros são impostores
        for genuine in biometric_reference.keys(): #Para cada usuario no banco
            genuine_data = valid_data.loc[valid_data['subject']==genuine].iloc[:,1:]
            impostor_df = valid_data.loc[valid_data['subject']!=genuine].iloc[:,1:]
            impostor_data  = impostor_df.sample(valid_size, replace=False, random_state=self.random_state).sort_index() #sorteio de 20 amostras aleatorias dentre todas as possíveis      

            if detector.name == 'M2005':
                score_type = 'similarity'
                genuine_scores = genuine_data.apply(lambda x: detector.score(x, biometric_reference[genuine].model), axis=1) 
                impostor_scores = impostor_data.apply(lambda x: detector.score(x, biometric_reference[genuine].model), axis=1)
            else:
                Exception("Choose a valid model.")
            
            decision_threshold = self.calculate_best_threshold(user_scores=genuine_scores, impostor_scores=impostor_scores, score_type=score_type)
            thresholds[genuine] = decision_threshold
        return thresholds

    def calculate_best_threshold(self, user_scores, impostor_scores, score_type):
        predictions = list(user_scores) + list(impostor_scores)
        labels = np.concatenate((np.ones(len(user_scores)), np.zeros(len(impostor_scores))))
        best_bacc = -float("inf")
        for score in predictions:
            _, _, bacc = self.reporter(scores=predictions, true_labels=labels, threshold=score, score_type=score_type)
            if bacc > best_bacc:
                best_bacc = bacc
                decision = score
        return decision

    def reporter(self, scores, true_labels, threshold, score_type):
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
