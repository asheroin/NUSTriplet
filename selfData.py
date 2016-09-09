
import numpy as np


class Tensor(object):
	"""docstring for Tensor"""
	def __init__(self, shape,data=None,rand_init=True):
		super(Tensor, self).__init__()
		self.shape = shape
		if data!=None:
			self.data = data.copy()
		elif rand_init==True:
			self.data = np.random.rand(shape[0]*shape[1]*shape[2])
		else:
			self.data = np.zeros(shape[0]*shape[1]*shape[2])

	def __getitem__(self,ind):
		return self.data[ind[0]*self.shape[1]*self.shape[2]+ind[1]*self.shape[2]+ind[2]]
	def __str__(self):
		return str(self.data)+'\n'+ "size="+str(self.shape)
	def __repr__ (self):
		return str(self.data)+'\n'+ "size="+str(self.shape)






class tripletDict(object):
	"""docstring for tripletDict"""
	def __init__(self,dim1,dim2,dim3,value,shape):
		super(tripletDict, self).__init__()
		self.arg = arg
		
		if len(dim1)!=len(dem2):
			# check input
			# should be more conditions
			pass
		if len(shape)!=3:
			pass
		


class triplet():
	"""docstring for triplet"""
	def __init__(self,dim1,dim2,dim3,value,shape):
		if len(dim1)!=len(dem2):
			# check input
			# should be more conditions
			pass
		if len(shape)!=3:
			pass

		self.__data_len = len(value)



		self.dim1 = dim1.copy()
		self.dim2 = dim2.copy()
		self.dim3 = dim3.copy()

		self.data = np.zeros(self.__data_len)
		self.index = np.zeros(self.__data_len)
		self.indptr = np.zeros(shape[0]*shape[1]+1)

		self.data = value.copy()
		self.index = dim3.copy()

		'''initialize indptr1 and indptr2'''
		# wait to write
		self.__init_data__()
	def __init__data__(self):
		tmp = np.zeros(shape[0]*shape[1])
		for a,b in zip(self.dim1,self.dim2):
			tmp[a*shape[0]]

	def __getitem__(self,ind):
		ind1 = ind[0]
		ind2 = ind[1]
		ind3 = ind[3]
		if ():
			# check dimention
			pass
		image_index_list = self.indptr1[ind1:ind1+1]
		image_list = self.indptr2[image_index_list]
		tag_list = self.index[image_list[ind2:ind2+1]]
		if ind3 in tag_list:
			offset = tag_list.index(ind3)
			return self.data[self.indptr2[self.inptr1[ind1]]+offset]
		else:
			return 0