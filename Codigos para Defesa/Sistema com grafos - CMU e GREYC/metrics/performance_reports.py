from _mypath import rootpath

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix


def eval_adaptive_methods(recommended, ideal, labels):
    if len(recommended) != len(ideal):
        raise Exception("Tamanhos diferentes!")
        
    cm = confusion_matrix(y_true=[*ideal.values()], y_pred=[*recommended.values()],labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
        
    count = 0
    for key in recommended.keys():
        if recommended[key] == ideal[key]:
            count=count+1
    print(f"Acurácia: {count / len(recommended)} ({count} / {len(recommended)})")
    print()
    
def get_df_metrics(metrics, adaptive_methods):
    """ 
        metrics :: ideal(rec)_metrics
        adaptive_methods :: ideal(rec)_adaptives
    """
    temp = dict()
    for user in metrics.keys():
        for metric in metrics[user].keys(): 
            temp.setdefault(user,{}).setdefault(metric, np.mean(metrics[user][metric]))
            #print(np.mean(rec_metrics0[user][metric]))
    df_recommended = pd.DataFrame(temp).T
    df_recommended = df_recommended.join(pd.DataFrame.from_dict(adaptive_methods,orient='index',columns=['rec_adaptive_method']))
    return df_recommended

def extract_mean_values(dicionary):
    """ Extrai a média dos valores de um dicionário de dicionários."""
    new = dict()
    for key1 in dicionary:
        for key2 in dicionary[key1]:
            new.setdefault(key1, {}).setdefault(key2, np.mean(dicionary[key1][key2]))
    return new

def get_table(confusion_matrix=True):
    rec_metrics0 = pickle.load(open(os.path.join(rootpath,"results/metricas/rec_metrics_fold"+str(0)+".pk"),"rb"))
    ideal_metrics0 = pickle.load(open(os.path.join(rootpath,"results/metricas/ideal_metrics_fold"+str(0)+".pk"),"rb"))
    rec_adaptives0 = pickle.load(open(os.path.join(rootpath,"results/metricas/rec_adaptives_fold"+str(0)+".pk"),"rb"))
    ideal_adaptives0 = pickle.load(open(os.path.join(rootpath,"results/metricas/ideal_adaptives_fold"+str(0)+".pk"),"rb"))

    rec_metrics1 = pickle.load(open(os.path.join(rootpath,"results/metricas/rec_metrics_fold"+str(1)+".pk"),"rb"))
    ideal_metrics1 = pickle.load(open(os.path.join(rootpath,"results/metricas/ideal_metrics_fold"+str(1)+".pk"),"rb"))
    rec_adaptives1 = pickle.load(open(os.path.join(rootpath,"results/metricas/rec_adaptives_fold"+str(1)+".pk"),"rb"))
    ideal_adaptives1 = pickle.load(open(os.path.join(rootpath,"results/metricas/ideal_adaptives_fold"+str(1)+".pk"),"rb"))

    rm0 = extract_mean_values(rec_metrics0)
    rm1 = extract_mean_values(rec_metrics1)
    im0 = extract_mean_values(ideal_metrics0)
    im1 = extract_mean_values(ideal_metrics1)

    ideal_adaptives0 = ideal_adaptives0.set_index('subject').T.to_dict('records')[0]
    ideal_adaptives1 = ideal_adaptives1.set_index('subject').T.to_dict('records')[0]

    if confusion_matrix:
        print("Fold 0")
        eval_adaptive_methods(rec_adaptives0, ideal_adaptives1, labels=['False','SlidingWindow','GrowingWindow'])

        print("Fold 1")
        eval_adaptive_methods(rec_adaptives1, ideal_adaptives0, labels=['False','SlidingWindow','GrowingWindow'])

    d1 = {user:1 if rec_adaptives0[user]==ideal_adaptives1[user] else 0 for user in rec_adaptives0.keys()}
    d2 = {user:1 if rec_adaptives1[user]==ideal_adaptives0[user] else 0 for user in rec_adaptives1.keys()}
    d3 = pd.concat([pd.DataFrame.from_dict(d1, orient="index", columns=["res"]), pd.DataFrame.from_dict(d2, orient="index", columns=["res"])])

    df_recommended0 = get_df_metrics(metrics=rm0, adaptive_methods=rec_adaptives0)
    df_recommended1 = get_df_metrics(metrics=rm1, adaptive_methods=rec_adaptives1)
    df_recommended = pd.concat([df_recommended0, df_recommended1]).sort_index()

    df_ideal = pd.concat([pd.DataFrame(im0), pd.DataFrame(im1)]).sort_index()
    df_ideal['ideal_adaptive_method'] = df_ideal.idxmax(axis=1)

    df_full = pd.merge(df_ideal, df_recommended, left_index=True, right_index=True)
    df_full = df_full.join(d3)
    df_full = df_full.reset_index().rename(columns={'index' : 'subject'})

    return df_full