# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime

import numpy as np
import random as rd 
from selfData import *
from joblib import Parallel, delayed

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

ISOTIMEFORMAT='%Y-%m-%d %X'

class coTriplet():
	"""docstring for coTriplet"""
	def __init__(self, **kwargs):
		self.UCore = 50
		self.ICore = 250
		self.TCore = 50
		# self._parse_kwasrgs(**kwargs)
	def _parse_kwargs(self,**kwargs):
		'''
		Initialize the Model hyperparameters
		'''
		# self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
	# 统计原始数据信息，并创建映射函数
	def _creat_dict2num(self,InputData):
		self.id2user = list(InputData.groupby('user').size().index)
		self.id2image = list(InputData.groupby('photo').size().index)
		self.id2tag = list(InputData.groupby('tag').size().index)

		#
		self.unique_u_num = len(self.id2user)
		self.unique_i_num = len(self.id2image)
		self.unique_t_num = len(self.id2tag)
		#
		self.user2id = dict((uid,i) for (i,uid) in enumerate(self.id2user))
		self.image2id = dict((pid,i) for (i,pid) in enumerate(self.id2image))
		self.tag2id = dict((tid,i) for (i,tid) in enumerate(self.id2tag))
		pass	
	def _init_params(self):
		pass
		# Initialize latent parameters
		# self.core =
		self.core = Tensor([self.UCore,self.ICore,self.TCore],rand_init=True)
		self.ucore = np.random.rand(self.unique_u_num,self.UCore) 
		self.icore = np.random.rand(self.unique_i_num,self.ICore) 
		self.tcore = np.random.rand(self.unique_t_num,self.TCore)
		self.alpha = 0.1
		self.gamma = 0.1
		self.gamma_c = 0.1 
		self.max_iter = 10
	def _item2id(self,InputData):
		uid = map(lambda x:self.user2id[x],InputData['user'])
		pid = map(lambda x:self.image2id[x],InputData['photo'])
		tid = map(lambda x:self.tag2id[x],InputData['tag'])
		InputData['uid'] = uid
		InputData['pid'] = pid
		InputData['tid'] = tid
		return InputData[['uid','pid','tid']]

	def fit(self,InputData):
		'''
		数据被设定为可以通过[]获取，如X[1,2,3]
		'''
		#self.rawData = InputData

		self._creat_dict2num(InputData)
		idData = self._item2id(InputData)
		self._init_params()
		self._update(idData)
		wait()
	def _update(self,InputData):
		'''

		'''
		for itertime in xrange(self.max_iter):
			for u in xrange(self.unique_u_num):
				X_tmp = InputData[InputData.uid==u]
				unique_i = X_tmp[['pid']].groupby('pid',as_index=False).size().index
				for i in unique_i:
					print 'processing uid=%s/image=%d|'%(self.id2user[u],self.id2image[i]),
					pos_tag,neg_tag = self._get_pos_and_neg_tag(InputData,u,i)
					z = 1/(len(pos_tag)*len(neg_tag))
					print 'got positive & negative tag,',
					y = self._cal_y(u,i)
					v = self._cal_v(y,pos_tag,neg_tag)
					print 'begin updata tensor...',
					self._update_factors(pos_tag,neg_tag,u,i,z,y,v)
					print 'finished'
	
	def _get_pos_and_neg_tag(self,InputData,u,i):
		a = InputData[InputData.uid==u]
		a = a[a.pid==i]
		a = a[['tid']].groupby('tid',as_index=False).size().index
		
		pos_tag = list(a)
		neg_tag = list(set(range(self.unique_t_num))-set(pos_tag))
		neg_tag = rd.sample(neg_tag,20)
		return pos_tag,neg_tag
	def _update_factors(self,pos_tag,neg_tag,u,i,z,y,v):
		self.core.data = self._update_para_core(u,i,z,y,v)
		self.ucore[u,0:self.UCore] = self._update_para_user(u,i,z,y,v)
		self.icore[i,0:self.ICore] = self._update_para_image(u,i,z,y,v)
		# '''更新tag matrix的准备工作'''
		q = self._cal_q(u,i)
		for tp in pos_tag:
			self.tcore[tp,0:self.TCore] = self._update_para_pos_tag(u,i,z,y,v,q,tp,neg_tag)
		for tn in neg_tag:
			self.tcore[tn,0:self.TCore] = self._update_para_neg_tag(u,i,z,y,v,q,tn,pos_tag)
	def _cal_y(self,u,i):
		res = np.tensordot(self.core.data.reshape(self.UCore,self.ICore,self.TCore),self.ucore[u,0:self.UCore].T,(0,0))
		res = np.tensordot(res,self.icore[i,0:self.ICore].T,(0,0))
		y = np.tensordot(res,self.tcore,(0,1))
		return y
	def _cal_v(self,y,pos_tag,neg_tag):
		v = np.arange(self.TCore)
		for tp in pos_tag:
			for tn in neg_tag:
				s = 1/(1+np.exp(-y[tp]+y[tn]))
				w = s*(1-s)
				for vtc in range(self.TCore):
					v[vtc] +=w*(self.tcore[tp,vtc]-self.tcore[tn,vtc])
		return v
	def _cal_q(self,u,i):
		tmp = np.tensordot(self.core.data.reshape(self.UCore,self.ICore,self.TCore),self.ucore[u,0:self.UCore],(0,0))
		tmp = np.tensordot(tmp,self.icore[i,0:self.ICore],(0,0))
		return tmp
	def _update_para_core(self,u,i,z,y,v):
		core_index = [(uu,ii,tt) for uu in range(self.UCore) for ii in range(self.ICore) for tt in range(self.TCore)  ]
		
		r = np.zeros(len(core_index))
		for ind,index in enumerate(core_index):
			r[ind]=self._solve_para_core(u,i,z,y,v,index)
		return r
	def _solve_para_core(self,u,i,z,y,v,index,alpha=0.1,gamma_c=0.1):
		grad = z*self.ucore[u,index[0]]*self.icore[i,index[1]]*v[index[2]]
		'''gradient descent'''
		new = self.core[index[0],index[1],index[2]]+\
					alpha * (grad-gamma_c *self.core[index[0],index[1],index[2]] )
		return new
	def _update_para_user(self,u,i,z,y,v):
		tmp = np.tensordot(self.core.data.reshape(self.UCore,self.ICore,self.TCore),self.icore[i,0:self.ICore],(1,0))
		tmp = z*np.tensordot(tmp,v,(1,0))
		return tmp
	def _update_para_image(self,u,i,z,y,v):
		tmp = np.tensordot(self.core.data.reshape(self.UCore,self.ICore,self.TCore),self.ucore[u,0:self.UCore],(0,0))
		tmp = z*np.tensordot(tmp,v,(1,0))
		return tmp
	def _update_para_pos_tag(self,u,i,z,y,v,q,tp,neg_tag):
		res = np.zeros(self.TCore)
		for index,val in enumerate(range(self.TCore)):
			res[index] = self._solve_para_pos_tag(u,i,z,y,v,q,tp,neg_tag,index)
		return res
	def _solve_para_pos_tag(self,u,i,z,y,v,q,tp,neg_tag,index,alpha=0.1,gamma=0.1):
		grad = 0
		for tn in neg_tag:
			s = 1/(1+np.exp(-y[tp]+y[tn]))
			w = s*(1-s)
			grad+=w
		grad = -z*q[index]*grad
		new = self.tcore[tp,index]+alpha*(grad-gamma*self.tcore[tp,index])
		return new
	def _update_para_neg_tag(self,u,i,z,y,v,q,tn,pos_tag):
		res = np.zeros(self.TCore)
		for index,val in enumerate(range(self.TCore)):
			res[index] = self._solve_para_neg_tag(u,i,z,y,v,q,tn,pos_tag,index)
		return res
	def _solve_para_neg_tag(self,u,i,z,y,v,q,tn,pos_tag,index,alpha=0.1,gamma=0.1):
		grad = 0
		for tp in pos_tag:
			s = 1/(1+np.exp(-y[tp]+y[tn]))
			w = s*(1-s)
			grad+=w
		grad = z*q[index]*grad
		new = self.tcore[tn,index]+alpha*(grad-gamma*self.tcore[tn,index])
		return new

	def _update_theta():
		'''Update factor'''
	



