""" Base class Module where all layers inherit from"""
from pathlib import Path
import os 
import pickle 
import numpy as np 

class Module: 
    def __init__(self):
       self.training = True  
       self.state_dict = {} 
       self.parameters = []
    

    def save(self) ->None: 
        pass 

    def load(self) ->None: 
        pass 
    def train(self)->None: 
        """ set the module in training mode """
        self.training = True 

    def eval(self)->None: 
        """ set the module in evaluation mode """
        self.training = False 
    



    



