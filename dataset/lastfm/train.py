# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:32:13 2023

@author: simon
"""

import pandas as pd

#data -- for format['user_id', 'item_id', 'ratings/purchased_time']
def IO():
    df = pd.read_csv('ratings.txt', sep='\t', header=None)
    
    num_user = len(df[0].unique())
    num_item = len(df[1].unique())
    
    df = df[df[2]>=2]
    
    train = df.sample(frac=0.8)
    val = train.sample(frac=0.1)
    train = train.drop(val.index)
    test = df.drop(train.index)
    train = train.sort_values([train.columns[0], train.columns[2]], ascending=[True, False], ignore_index=True)
    val = val.sort_values([val.columns[0], val.columns[2]], ascending=[True, False], ignore_index=True)
    test = test.sort_values([test.columns[0], test.columns[2]], ascending=[True, False], ignore_index=True)
    
    return df, num_user, num_item, train, val, test

df, num_user, num_item, train, val, test = IO()