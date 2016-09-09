# -*- coding: utf-8 -*-


import coTriplet as ct 
import pandas as pd 


if __name__ == '__main__':
	solution = ct.coTriplet()
	data = pd.read_csv('photo-user-tag.csv')
	solution.fit(data)