#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:55:47 2018

@author: jcarraascootarola
"""

import unittest
from SigmoidNeuron import SigmoidNeuron 
from NeuralNetwork import NeuralNetwork
import numpy as np
 
class TestUM(unittest.TestCase):
 
    def setUp(self):
        pass
    
    def test0_create(self):
        a=NeuralNetwork(0.5,2,3,[4,5,1])
        
        self.assertEqual( len(a.layers),3 )
        self.assertEqual( len(a.layers[0].neurons),4 )
        self.assertEqual( len(a.layers[1].neurons),5 )
        self.assertEqual( len(a.layers[2].neurons),1 )
        self.assertEqual( len(a.layers[0].neurons[0].weights),2 )
        self.assertEqual( len(a.layers[1].neurons[0].weights),4 )
        self.assertEqual( len(a.layers[2].neurons[0].weights),5 )
 
    def test1_train(self):
        a=NeuralNetwork(0.5,2,2,[1,1])
        
        a.layers[0].neurons[0].weights=np.array([0.4,0.3])
        a.layers[0].neurons[0].bias=0.5
        a.layers[1].neurons[0].weights=np.array([0.3])
        a.layers[1].neurons[0].bias=0.4
        
        a.train([[1,1]],[[1]])
    
        self.assertEqual( list(a.layers[0].neurons[0].weights), [0.40210150899948904, 0.302101508999489])
        self.assertEqual( a.layers[0].neurons[0].bias, 0.502101508999489)
        self.assertEqual( list(a.layers[1].neurons[0].weights), [0.33030663725605847])
        self.assertEqual( a.layers[1].neurons[0].bias, 0.43937745312797394)
        
    def test2_train(self):
        a=NeuralNetwork(0.5,2,2,[2,2])
        
        a.layers[0].neurons[0].weights=np.array([0.7,0.3])
        a.layers[0].neurons[0].bias=0.5
        a.layers[0].neurons[1].weights=np.array([0.3,0.7])
        a.layers[0].neurons[1].bias=0.4
        a.layers[1].neurons[0].weights=np.array([0.2,0.3])
        a.layers[1].neurons[0].bias=0.3
        a.layers[1].neurons[1].weights=np.array([0.4,0.2])
        a.layers[1].neurons[1].bias=0.6
        
        a.train([[1,1]],[[1,1]])
        
        self.assertEqual( list(a.layers[0].neurons[0].weights), [0.7025104485493278, 0.3025104485493278])
        self.assertEqual( a.layers[0].neurons[0].bias, 0.5025104485493278)
        self.assertEqual( list(a.layers[0].neurons[1].weights), [0.30249801135748333, 0.7024980113574834])
        self.assertEqual( a.layers[0].neurons[1].bias, 0.40249801135748337)
        self.assertEqual( list(a.layers[1].neurons[0].weights), [0.22998842544522746, 0.3294270895020592])
        self.assertEqual( a.layers[1].neurons[0].bias, 0.3366295422515899)
        self.assertEqual( list(a.layers[1].neurons[1].weights), [0.41945668784741397, 0.21909248939212306])
        self.assertEqual( a.layers[1].neurons[1].bias,0.6237654881509048)
    
if __name__ == '__main__':
    unittest.main()