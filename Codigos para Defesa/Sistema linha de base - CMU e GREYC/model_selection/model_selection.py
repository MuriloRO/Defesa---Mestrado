from _mypath import rootpath

import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm
import IPython.display as ipd
import pickle
import os

from biometric_system import BiometricSystem
from data_stream.data_stream import Random

def get_combinations(users):
    """ Utiliza o método `KFold()` para separar a lista de usuários disponíveis em possíveis combinações
    de usuário genuíno (genuine), lista de usuários registrados no sistema (user_r) e lista de usuários 
    não registrados no sistema (user_nr). Cada combinação é salva e retornada em uma tupla no formato 
    (genuine, user_r, user_nr)."""

    #print("Criando possíveis combinações de usuário entre [genuíno, impostor_interno, impostor_externo].\n")
    
    # TRANSFORMAR SEQUENCE EM DICIONÁRIO
    kfold = KFold(n_splits=5)
    splits = kfold.split(users)
    sequence = list()
    for registered, not_registered in splits:
        user_r = users[registered].tolist()
        user_nr = users[not_registered].tolist()
        for genuine in user_r:
            sequence.append((genuine, user_r, user_nr))
    return sequence

class GridSearch:
    """ Implementação do algoritmo de busca de melhores parâmetros GridSearch voltada para o problema da pesquisa.
    
    Parâmetros
    ---
    detector :: Algoritmo de detecção para dinâmicas da digitação (pacote `anomaly_detectors`).
    parameters :: Parâmetros que serão testados no GridSearch. Para este problema testaremos apenas os diferentes métodos de adaptação disponíveis.
    
    """

    def __init__(self, detector, parameters, random_state):
        self.detector = detector
        self.parameters = ParameterGrid(parameters)
        self.random_state = random_state

    def fit(self, dataset, fold):
        """ Recebe um conjunto de dados `dataset` e calcula a acurácia balanceada de cada parâmetro. Cada usuário é testado 4x como 
        genuíno, em cada teste, um fluxo de dados diferente é testado. Os resultados obtidos em cada fold, para cada parâmetro, são 
        salvos em `results/metricas/ideal_metrics_fold`.
        """
        
        results = dict()
        users = dataset['subject'].unique()
        user_config = get_combinations(users)

        bsystem = BiometricSystem(detector=self.detector, use_metaknowledge=False, random_state=self.random_state)
        _, test_data = bsystem.enrollment(dataset=dataset,
                                n_samples=20,
                                user_column='subject')

        for j, params in enumerate(self.parameters):
            metrics = dict()
        
            for i, (genuine, user_r, user_nr) in enumerate(user_config):
                ipd.clear_output(wait=True)
                print(f"Extraindo metaconhecimento\nParamêtro: {j+1} / {len(self.parameters)}\nUsuário: {i+1} / {len(user_config)}")

                datastream = Random(impostor_rate=0.2, rate_external_impostor=0)
                test_stream, y_true = datastream.create(test_data, genuine, user_r, user_nr, self.random_state)
                y_pred = bsystem.autenticate(genuine, test_stream, adaptive=params['adaptive'], return_scores=False)
                fmr, fnmr, b_acc = bsystem.compute_metrics(y_true, y_pred)

                metrics.setdefault(genuine, []).append(b_acc)
            
            #user_scores = {key:np.mean(values) for key, values in metrics.items()}
            results[params['adaptive']] = metrics
        
        pickle.dump(results, open(os.path.join(rootpath,"results/metricas/ideal_metrics_fold"+str(fold)+".pk"),"wb"))

        return results