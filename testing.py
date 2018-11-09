#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:55:47 2018

@author: jcarraascootarola
"""

import unittest
from SigmoidNeuron import SigmoidNeuron 
from NeuralNetwork import NeuralNetwork

 
class TestUM(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test(self):
        self.assertEqual( multiply(3,4), 12)
 
    def test_strings_a_3(self):
        self.assertEqual( multiply('a',3), 'aaa')
 
if __name__ == '__main__':
    unittest.main()