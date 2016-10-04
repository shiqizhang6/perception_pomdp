#! /usr/bin/env python


import random
import os
import numpy as np
import sys
import csv
import copy

# classifiers
from sklearn import svm

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
	
	def getObjectIDs(self):
		return self._object_ids
	
	def isPredicateTrue(self,predicate,object_id):
		if predicate in object_id:
			return True
		return False
	
	
	def createScikitClassifier(self):
		# currently an SVM
		return svm.SVC(gamma=0.001, C=100.)
	
	def crossValidate(self,X,Y,num_tests):
		scores = []
		
		for fold in range(0,num_tests):
			# shuffle data
			random.seed(fold)
			X_f = copy.deepcopy(X)
			random.shuffle(X_f)
			
			random.seed(fold)
			Y_f = copy.deepcopy(Y)
			random.shuffle(Y_f)
			
			# split into train (2/3) and test (1/3)
			X_f_train = X_f[0:len(X_f)*2/3]
			Y_f_train = Y_f[0:len(Y_f)*2/3]
			
			X_f_test = X_f[len(X_f)*2/3:len(X_f)]
			Y_f_test = Y_f[len(Y_f)*2/3:len(Y_f)]
			
			# create and train classifier
			classifier_f = self.createScikitClassifier()
			classifier_f.fit(X_f_train, Y_f_train)
			
			
			score_f = classifier_f.score(X_f_test, Y_f_test) 
			scores.append(score_f)
		mean_score = np.mean(scores)
		print mean_score
		  
	
	def performCrossValidation(self, num_tests):
		for predicate in self._predicates:
			# this contains the context-specific classifier for the predicate
			classifier_ensemble_dict = self._predicate_classifier_dict[predicate]
			
			# this contains the data for the predicates
			pred_data_dict = self._predicate_data_dict[predicate]
			
			for context in self._contexts:
				[X,Y] = pred_data_dict[context]
				print("Cross-validating predicate " + predicate + " and context "+context+" with " + str(len(X)) + " points")
				self.crossValidate(X,Y,num_tests)
	
	# train_objects: a list of training objects
	# num_interaction_trials:	how many datapoint per object, from 1 to 10
	def trainClassifiers(self,train_objects,num_interaction_trials):
		# for each predicate
		
		# dictionary storing the ensemble of classifiers (one per context) for each predicate
		self._predicate_classifier_dict = dict()
		self._predicate_data_dict = dict()
		
		for predicate in self._predicates:
			
			# dictionary storing the classifier for each context for this predicate
			classifier_p_dict = dict()
			data_p_dict = dict()
			
			print("Training classifiers for predicate '"+predicate+"'")
			# separate positive and negative examples
			positive_object_examples = []
			negative_object_examples = []
			for o in train_objects:
				if self.isPredicateTrue(predicate,o):
					positive_object_examples.append(o)
				else:
					negative_object_examples.append(o)
			
			if len(positive_object_examples) == 0 or len(negative_object_examples) == 0:
				print("[WARN] skipping training as either positive or negative examples are not available")
				continue
			
			print("Positive examples: "+str(positive_object_examples))
			print("Negative examples: "+str(negative_object_examples))
			
			# train classifier for each context
			for context in self._contexts:
				# create dataset for this context 
				X = []
				Y = []
				for o in positive_object_examples:
					for t in range(1,num_interaction_trials+1):
						x_ot = self.getFeatures(context,o,t)
						y_ot = 1
						X.append(x_ot)
						Y.append(y_ot)
				for o in negative_object_examples:
					for t in range(1,num_interaction_trials+1):
						x_ot = self.getFeatures(context,o,t)
						y_ot = 0
						X.append(x_ot)
						Y.append(y_ot)
				
				# the dataset is now ready; X is the inputs and Y the outputs or target
				
				# create the SVM
				print("Training classifier with "+str(len(X)) + " datapoints.")
				classifier_cp = self.createScikitClassifier()
				classifier_cp.fit(X, Y)
				
				# store the classifier and the dataset
				classifier_p_dict[context] = classifier_cp
				data_p_dict[context] = [X,Y]  
				  
				#print("Dataset for context "+context+":")
				#print X
				#print Y
			
			# store ensemble in dictionary
			self._predicate_classifier_dict[predicate] = classifier_p_dict
			self._predicate_data_dict[predicate] = data_p_dict
		
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
	
	# some train parameters
	num_train_objects = 24
	num_trials_per_object = 10
	
	
	# get all object ids and shuffle them
	object_ids = copy.deepcopy(classifier.getObjectIDs());
	
	random.seed(1)
	random.shuffle(object_ids)
	print object_ids

	
	# do it again to check that the random seed shuffles the same way
	#object_ids2 = classifier.getObjectIDs();
	#random.seed(1)
	#random.shuffle(object_ids2)
	#print object_ids2
	
	
	# pick random subset for train
	train_object_ids = object_ids[0:num_train_objects]
	#print train_object_ids
	
	# train classifier
	classifier.trainClassifiers(train_object_ids,num_trials_per_object)
	
	
	
	# perform cross validation to figure out context specific weights for each predicate (i.e., the robot should come up with a number for each sensorimotor context that encodes how good that context is for the predicate
	classifier.performCrossValidation(10)
	
	
if __name__ == "__main__":
    main(sys.argv)
