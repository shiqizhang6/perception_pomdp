#! /usr/bin/env python


import random
import os
import numpy as np
import sys
import csv

class ClassifierICRA(object):
	
	def __init__(self, data_path, behaviors, modalities,predicates):
		# load data
		self._path = data_path
		self._behaviors = behaviors
		self._modalities = modalities
		self._predicates = predicates
		
		# compute lists of contexts
		self._contexts = []
		
		for b in behaviors:
			for m in modalities:
				if self.isValidContext(b,m):
					context_bm = b+"-"+m
					self._contexts.append(context_bm)
		
		# print to verify
		print "Valid contexts:"
		print self._contexts
		
		# load object ids
		self._object_ids = []
		object_file = self._path +"/objects.txt"
		with open(object_file, 'rb') as f:
			reader = csv.reader(f)
			for row in reader:
				self._object_ids.append(row[0])
				
		print "Set of objects:"
		print self._object_ids
		
		# load data for each context
		
		# dictionary holding all data for a given context (the data is a dictionary itself)
		self._context_db_dict = dict()
		
		for context in self._contexts:
			context_filename = self._path+"/sensorimotor_features/"+context+".txt"
			
			# count how many datapoint we've seen with each object
			object_trial_count_dict = dict()
			for o in self._object_ids:
				object_trial_count_dict[o]=0
			
			# dictionary holding all data in this context
			# key: "<object_id>_<trial_integer>" (e.g., "heavy_blue_glass_4")
			# data: feature vector of floats
			data_dict = dict()
			
			print('Loading ' + context_filename+ '...')
			with open(context_filename, 'rb') as f:
				reader = csv.reader(f)
				for row in reader:
					obj = row[0]
					features = row[1:len(row)]
					object_trial_count_dict[obj] += 1
					
					key = obj+"_"+str(object_trial_count_dict[obj])
					data_dict[key] = features
					
			
			self._context_db_dict[context] = data_dict
	
	def getFeatures(self,context,object_id,trial_number):
		key = object_id+"_"+str(trial_number)
		return self._context_db_dict[context][key]
	
	def isPredicateTrue(self,predicate,object_id):
		if predicate in object_id:
			return True
		return False
	
	# train_objects: a list of training objects
	# num_interaction_trials:	how many datapoint per object, from 1 to 10
	def trainClassifier(self,train_objects,num_interaction_trials):
		# for each predicate
		for predicate in self._predicates:
			
			# separate positive and negative examples
			positive_examples = []
			negative_examples = []
			for o in train_objects:
				if self.isPredicateTrue(predicate,o):
					positive_examples.append(o)
				else:
					negative_examples.append(o)
			
			# train classifier for each context
	
	def isValidContext(self,behavior,modality):
		if behavior == "look":
			if modality == "color" or modality == "patch":
				return True
			else: 
				return False
		elif modality == "proprioception" or modality == "audio":
			return True
		else:
			return False
		

def main(argv):
		
	datapath = "../data/icra2014"
	behaviors = ["look","grasp","lift_slow","hold","shake","high_velocity_shake","low_drop","tap","push","poke","crush"]
	modalities = ["color","patch","proprioception","audio"]

	predicates = ['brown','green','blue','light','medium','heavy','glass','screws','beans','rice']


	classifier = ClassifierICRA(datapath,behaviors,modalities,predicates)
	print "Classifier created..."
	
	#test_f = classifier.getFeatures("look-color","light_brown_glass",10)
	#print test_f
	
	
if __name__ == "__main__":
    main(sys.argv)
