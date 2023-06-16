# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:12:12 2023

@author: ppras
"""

import numpy as np
import pickle
import streamlit as st
#loading saved model
loaded_model=pickle.load(open("Breast_cancer_data.sav",'rb'))
#creating function for prediction
def Breast_cancer_pedicition(input_data):

        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        
        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)
        
        if (prediction[0] == 0):
           return 'The person has no breast cancer'
        else:
          return 'The person has breast cancer'
def main():
          #Giving the table
         st.title('Breast cancer Prediction Web Page')
          
          #Making InputData Web Page for Enduser
          
         mean_radius= st.text_input('mean radius')
         mean_texture=st.text_input('Mean Texture')
         mean_perimeter=st.text_input('Mean Perimeter')
         mean_area=st.text_input('Mean Area')
         mean_smoothness=st.text_input('Mean Smoothness')
          
         #Code for Prediction
         daignosis=""
          
          #Creating Examine for Prediction
         if st.button('BreasrcancerTestResult'):
              daignosis=Breast_cancer_prediction([mean_radius, mean_texture,mean_perimeter,mean_area,mean_smoothness])
              
         st.success(daignosis)
          
if __name__=='__main__':
    main()
