# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:59:38 2019

@author: mh iyer
"""

import os
import sys
import numpy as np
import math
import pandas as pd

eps  = 0.0000000001
    
def calc_entropy(col):

    # num 0 counts
    num_zeros = sum([True if x==0 else False for x in col])
    # num 1 counts
    num_ones = sum([True if x==1 else False for x in col])
    # probability of 0
    prob_0 = (num_zeros*1.)/(num_zeros + num_ones)
    # probability of 1
    prob_1 = 1. - prob_0
    
    entropy = -1* (prob_0* math.log2(prob_0+eps) + prob_1*math.log2(prob_1+eps))
    return(entropy)


def InfoGain(H_A,df,attribute,target):
    # H_A is the entropy of the class
    # df is the entire dataframe
    infogain_a_b = H_A
    
    # get pandas series of attribute
    B = df[attribute]
    
    # get unique values
    unique_values = B.unique()
    unique_value_counts = B.value_counts()
    
    # loop through the unique values, e.g. as 'Sky' has two unique values 'Sunny' and 'Rainy'- these are looped.
    for unique_value in unique_values:
            
        # probability of attribute- add eps to avoid math domain error
        num_occurrences = unique_value_counts[unique_value]    
        prob_unique_value = (num_occurrences+eps)/(len(B)+eps)
        
        # slice the column to get the corresponding target values
        target_given_B_is_unique_value = df[df[attribute]==unique_value][target]
        
        # get entropy
        if len(target_given_B_is_unique_value)>0:
            entropy_target_B_is_unique_value  = calc_entropy(target_given_B_is_unique_value)
        else:
            entropy_target_B_is_unique_value  = 0

    
        # update InfoGain
        infogain_a_b = infogain_a_b - (prob_unique_value*entropy_target_B_is_unique_value )
    
    return infogain_a_b


# main routine
if __name__ == "__main__":
    
    # define the data
    # example taken from Mitchell Chapter 2/3
    data = pd.DataFrame({"Sky":['Sunny','Sunny','Rainy','Sunny'],
                         "AirTemp":['Warm','Warm','Cold','Warm'],
                         "Humidity":['Normal','High','High','High'],
                         "Wind":['Strong','Strong','Strong','Strong'],
                         "Water":['Warm','Warm','Cool','Cool'],
                         "Forecast":['Same','Same','Change','Change'],
                         "EnjoySport":['Yes','Yes','No','Yes']})

    # define target column with values 0 or 1
    data['target'] = [1 if x=='Yes' else 0 for x in data['EnjoySport']]
    
    # define list of attributes to check- can be modified if required
    attributes = ['Sky','AirTemp','Humidity','Wind','Water','Forecast']
    
    
    ##################################
    #Get root node    
    ##################################
    
    # define parameters
    max_attribute_val = -10000
    info_dict = {}

    # get entropy of class
    H_A = calc_entropy(data['target'])
    
    # loop through attributes to find the one with the highest infogain
    for attribute in attributes:
        print('Analyzing:',attribute)
        infogain_current = InfoGain(H_A, data, attribute, 'target')
        if infogain_current>max_attribute_val:
            max_attribute_val = infogain_current
            node = attribute
            
        info_dict[attribute] = infogain_current
    
    # print the output
    print('Dictionary of Infogains across attributes:')
    print(info_dict)
    print('Node should be :',node,' with infogain of :',max_attribute_val)

