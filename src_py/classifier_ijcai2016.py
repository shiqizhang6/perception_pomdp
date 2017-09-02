#! /usr/bin/env python


import random
import os
import numpy as np
import sys
import csv
import copy

import pprint
import pickle

# classifiers
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import normalize

from oracle_full_ijcai2016 import TFTable

class ClassifierIJCAI(object):
	
	def __init__(self, data_path, behaviors, modalities, T_oracle, objects_ids_filename):
		# load data
		self._path = data_path
		self._behaviors = behaviors
		self._modalities = modalities
		self._T_oracle = T_oracle
		self._predicates = T_oracle.getAllPredicates()
		
		# predicates for which we have classifiers
		self._learned_predicates = []
		
		# some constants
		self._num_trials_per_object = 5
		self._train_test_split_fraction = 2/3 # what percentage of data is used for training when doing internal cross validation on training data
		
		# compute lists of contexts
		self._contexts = []
		
		for b in behaviors:
			for m in modalities:
				if self.isValidContext(b,m):
					context_bm = b+"_"+m
					self._contexts.append(context_bm)
		
		# print to verify
		#print("Valid contexts:")
		#print(self._contexts)
		
		# dictionary that holds context specific weights for each predicate
		self._pred_context_weights_dict = dict()
		
		# load object ids -- integers from 1 to 32
		self._object_ids = []
		for i in range(1,33):
			self._object_ids.append(i)
		
		self._object_name_dict = dict()
		
		# load string ids to map to ingeters
		string_to_int_id_dict = dict()
		with open(objects_ids_filename, 'r') as f:
			reader = csv.reader(f)
			for row in reader:
				string_id = str(row[0])
				int_id = int(row[1])
				#print(string_id+" "+str(int_id))
				string_to_int_id_dict[string_id]=int_id
				self._object_name_dict[int_id]=string_id
		
		#object_file = self._path +"/objects.txt"
		#with open(object_file, 'rb') as f:
		#	reader = csv.reader(f)
		#	for row in reader:
		#		self._object_ids.append(row[0])	
				
		#print("Set of objects:")
		#print(self._object_ids)
		
		# load data for each context
		
		# dictionary holding all data for a given context (the data is a dictionary itself)
		self._context_db_dict = dict()
		
		for context in self._contexts:
			context_filename = self._path+"/"+context+".txt"
			
			# count how many datapoint we've seen with each object
			object_trial_count_dict = dict()
			for o in self._object_ids:
				object_trial_count_dict[o]=0
			
			#print(object_trial_count_dict)
			
			# dictionary holding all data in this context
			# key: "<object_id>_<trial_integer>" (e.g., "heavy_blue_glass_4")
			# data: feature vector of floats
			data_dict = dict()
			
			#print('Loading ' + context_filename+ '...')
			with open(context_filename, 'r') as f:
				reader = csv.reader(f)
				for row in reader:
					
					obj = int(row[0])
					features = row[1:len(row)]
					object_trial_count_dict[obj] += 1
						
					key = str(obj)+"_"+str(object_trial_count_dict[obj])
					data_dict[key] = features
					
			
			self._context_db_dict[context] = data_dict
	
	def getFeatures(self,context,object_id,trial_number):
		key = str(object_id)+"_"+str(trial_number)
		return self._context_db_dict[context][key]
	
	def getObjectIDs(self):
		return self._object_ids
	
	def isPredicateTrue(self,predicate,object_id):
		return self._T_oracle.getTorF(predicate,str(object_id))
		#if predicate in object_id:
		#	return True
		#return False
	
	def getObjectName(self,object_id):
		return self._object_name_dict[object_id]
	
	def computeKappa(self,confusion_matrix):
		# compute statistics for kappa
		TN = confusion_matrix[0][0]
		TP = confusion_matrix[1][1]
		FP = confusion_matrix[1][0]
		FN = confusion_matrix[0][1]
		
		total_accuracy = (TN+TP)/np.sum(confusion_matrix)
		random_accuracy = ((TN+FP)*(TN+FN) + (FN+TP)*(FP+TP)) / ( np.sum(confusion_matrix) * np.sum(confusion_matrix))
		
		kappa = (total_accuracy - random_accuracy) / (1.0 - random_accuracy)
		return kappa
	
	# inputs: learn_prob_model (either True or False)
	def createScikitClassifier(self, learn_prob_model):
		# SVM
		#return svm.SVC(gamma=0.001, C=100, probability = learn_prob_model)
	
		#return svm.SVC(kernel="poly",C=10,degree=2,probability = learn_prob_model)
	
		# decision tree
		return tree.DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=None, min_samples_split=2, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
	
	
	
	def crossValidateObjectBased(self, predicate, context):
		
		# confusion matrix
		CM_total = np.zeros( (2,2) )
		
		# for each test object
		for test_object in self._train_objects:
			train_objects = []
			for o in self._train_objects:
				if o != test_object:
					train_objects.append(o)
			#print("\ttrain: "+str(train_objects)+"\ttest:"+str(test_object))
			X_train = []
			Y_train = []
			
			# create X and Y for train
			num_train_positive = 0
			num_train_negative = 0
			for o_train in train_objects:
				
				# get class label
				y_o = 0
				if self.isPredicateTrue(predicate,o_train):
					y_o = 1
					num_train_positive += 1
				else:
					num_train_negative += 1
				
				# for each trial, make datapoint
				for t in range(1,self._num_interaction_trials+1):
					x_ot = self.getFeatures(context,o_train,t)
					X_train.append(x_ot)
					Y_train.append(y_o)
			
			#print("\t# datapoints: "+str(len(X_train)))
	
			# check that there are two classes
			if num_train_positive == 0 or num_train_negative == 0:
				return 0.5 # default weight
					
			# create test data
			X_test = []
			Y_test = []
			y_t = 0
			if self.isPredicateTrue(predicate,test_object):
				y_t = 1 
			for t in range(1,self._num_interaction_trials+1):
				x_ot = self.getFeatures(context,test_object,t)
				X_test.append(x_ot)
				Y_test.append(y_t)
			
			#print("\t# test datapoints: "+str(len(X_test)))
	
			
			# create and train classifier
			classifier_t = self.createScikitClassifier(False)	
			classifier_t.fit(X_train, Y_train)
				
			# test classifier
			# for each test point
			Y_est = classifier_t.predict(X_test)
			
			# confusion matrix
			CM = np.zeros( (2,2) )
		
			for i in range(len(Y_est)):
				actual = Y_test[i]
				predicted = Y_est[i]
				
				CM[predicted][actual] = CM[predicted][actual] + 1
				
			CM_total = CM_total + CM
		return CM_total
	
	def crossValidate(self,X,Y,num_tests):
		scores = []
		
		# confusion matrix
		CM_total = np.zeros( (2,2) )
			
		
		for fold in range(0,num_tests):
			# shuffle data - both inputs and outputs are shuffled using the same random seed to ensure correspondance
			random.seed(fold)
			X_f = copy.deepcopy(X)
			random.shuffle(X_f)
			
			random.seed(fold)
			Y_f = copy.deepcopy(Y)
			random.shuffle(Y_f)
			
			# split into train (2/3) and test (1/3)
			X_f_train = X_f[0:int(len(X_f)*2/3)]
			Y_f_train = Y_f[0:int(len(Y_f)*2/3)]
			
			X_f_test = X_f[int(len(X_f)*2/3):len(X_f)]
			Y_f_test = Y_f[int(len(Y_f)*2/3):len(Y_f)]
			
			# create and train classifier
			classifier_f = self.createScikitClassifier(True)
			classifier_f.fit(X_f_train, Y_f_train)
			
			
			score_f = classifier_f.score(X_f_test, Y_f_test) 
			scores.append(score_f)
			
			# for each test point
			Y_f_est = classifier_f.predict(X_f_test)
			
			# confusion matrix
			CM = np.zeros( (2,2) )
			
			#print(Y_f_est)
			#print(Y_f_test)
			
			for i in range(len(Y_f_est)):
				#print(str(Y_f_est[i])+" "+str(Y_f_test[i]))
				actual = Y_f_test[i]
				predicted = Y_f_est[i]
				
				CM[predicted][actual] = CM[predicted][actual] + 1
				#if actual == 1 and predicted == 1:
				#	CM[1][1] = CM[1][1]+1
				#else if actual == 0 and predicted == 0:
				#	CM[0][0] = CM[0][0]+1
				#else if actual == 1 and predicted == 0:
				#	CM[0]
			#print(CM)
			CM_total = CM_total + CM
			
		#print(CM_total)
		#print(np.sum(CM_total))
		
		kappa = self.computeKappa(CM_total)

		return kappa
		
	def getPredicateBehaviorObservatoinModel(self,predicate,behavior):
		# confusion matrix
		CM = np.zeros( (2,2) )
		
		b_contexts = []
		for context in self._contexts:
			if behavior in context:
				b_contexts.append(context)
		
		for context in b_contexts:
			CM_c = self._pred_context_cm_dict[predicate][context]
			CM = CM + (self._pred_context_weights_dict[predicate][context] *CM_c)
		
		cm_sum = np.sum(CM)
		#print(cm_sum)	
		CM = CM / cm_sum
		return CM 
	
	def performCrossValidation(self, test_predicates):
		self._pred_context_cm_dict = dict()
		
		for predicate in test_predicates:
			
			self._pred_context_cm_dict[predicate] = dict()
			
			print("Cross-validating classifiers for "+predicate)
			# this contains the context-specific classifier for the predicate
			
			# check if classifier for predicate exists -- it may be possible that the set of training objects only contain positive or only contain negative examples in which case the classifier wouldn't have been created
			if predicate in self._predicate_classifier_dict.keys():
			
				classifier_ensemble_dict = self._predicate_classifier_dict[predicate]
			
				# this contains the data for the predicates
				pred_data_dict = self._predicate_data_dict[predicate]
			
				pred_context_weights = dict()
				#print("Predicate = "+predicate)
				for context in self._contexts:
					# perform object-based cross validation
					CM_ocv = self.crossValidateObjectBased(predicate,context)
					kappa_ocv = self.computeKappa(CM_ocv)
					
					# store CM
					self._pred_context_cm_dict[predicate][context] = CM_ocv
					
					#print(CM_ocv)
					#print("Obj. cross validation for "+predicate+" and "+context+":\t"+str(kappa_ocv))
					
					#[X,Y] = pred_data_dict[context]
					#print("Cross-validating predicate " + predicate + " and context "+context+" with " + str(len(X)) + " points")
					#kappa = self.crossValidate(X,Y,num_tests)
				
					# store the weight for that context and predicate
					pred_context_weights[context]=kappa_ocv
					
					if pred_context_weights[context] <= 0.0:
						pred_context_weights[context] = 0.001
					#else:
					#	print("\t"+context+":\t"+str(pred_context_weights[context]))
				
				self._pred_context_weights_dict[predicate] = pred_context_weights
			else:
				print("[WARN] Cannot perform CV for predicate "+predicate)
		#print("Context weight dict:")
		#print(self._pred_context_weights_dict)
	
	# train_objects: a list of training objects
	# num_interaction_trials:	how many datapoint per object, from 1 to 10
	def trainClassifiers(self,train_objects,num_interaction_trials, train_predicates):

			
		# dictionary storing the ensemble of classifiers (one per context) for each predicate
		self._predicate_classifier_dict = dict()
		self._predicate_data_dict = dict()
		
		# store train objects and num interaction trials
		self._train_objects = train_objects
		self._num_interaction_trials = num_interaction_trials
		
		for predicate in train_predicates:
			
			# dictionary storing the classifier for each context for this predicate
			classifier_p_dict = dict()
			data_p_dict = dict()
			
			# separate positive and negative examples
			positive_object_examples = []
			negative_object_examples = []
			for o in train_objects:
				if self._T_oracle.hasLabel(predicate,str(o)):
					if self.isPredicateTrue(predicate,o):
						positive_object_examples.append(o)
					else:
						negative_object_examples.append(o)
			
			
			
			if len(positive_object_examples) == 0 or len(negative_object_examples) == 0:
				#print("[WARN] skipping training as either positive or negative examples are not available")
				continue
			
			print("Training classifiers for predicate '"+predicate+"' with "+str(len(positive_object_examples)) +" positive and "+str(len(negative_object_examples))+" negative object examples.")
			
			
			#print("Positive examples: "+str(positive_object_examples))
			#print("Negative examples: "+str(negative_object_examples))
			
			# train classifier for each context
			for context in self._contexts:
				#print(context)
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
				#print("Training classifier with "+str(len(X)) + " datapoints.")
				classifier_cp = self.createScikitClassifier(True)
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
			
			# mark learned predicate
			self._learned_predicates.append(predicate)
	
	def learnedPredicates(self):
		return self._learned_predicates
	
	def isValidContext(self,behavior,modality):
		if modality == "surf":
			return True # all contexts have surf
		elif behavior == "look":
			if modality == "rgb" or modality == "color" or modality == "hsv" or modality == "fc7" or modality == "fpfh" or modality == "surf":
				return True
			else: 
				return False
		elif modality == "fingers":
			if behavior == "grasp":
				return True
			else:
				return False
		elif modality == "haptics" or modality == "audio" or modality == "corl-audio":
			return True
		else:
			return False
			
	# input: the target object, the behavior, and a predicate
	# output: the probability that the object matches the predicate		
	def classify(self, object_id, behaviors, predicate, selected_trial):
		
		# before doing anything, check whether we even have classifiers for the predicate
		if predicate not in self._predicate_classifier_dict.keys():
			# return negative result
			print("[WARNING] no classifier available for predicate "+predicate)
			return 0.0
			
			
		# first, randomly pick which trial we're doing
		num_available = self._num_trials_per_object
		if selected_trial == -1:
			selected_trial = random.randint(1,num_available)
		
		# next, find which contexts are available in that behavior
		b_contexts = []
		for context in self._contexts:
			for behavior in behaviors:
				if behavior in context:
					b_contexts.append(context)
		#print(b_contexts)
		#print(selected_trial)
		
		# call each classifier
		
		# output distribution over class labels (-1 and +1)
		classlabel_distr = [0.0,0.0]
		
		for context in b_contexts:
			
			# get the classifier for context and predicate
			classifier_c = self._predicate_classifier_dict[predicate][context]
			
			# get the data point for the object and the context
			x = self.getFeatures(context,object_id,selected_trial)
			
			# pass the datapoint to the classifier and get output distribuiton
			output = classifier_c.predict_proba([x])
			
			# weigh distribution by context reliability
			context_weight = 1.0
			
			# do this only if weights have been estimated
			if len(self._pred_context_weights_dict) != 0:
				context_weight = self._pred_context_weights_dict[predicate][context]
				
			#print(context_weight)
			
			classlabel_distr += context_weight*output[0]
			#print("Prediction from context "+context+":\t"+str(output))
		
		# normalize so that output distribution sums up to 1.0
		prob_sum = sum(classlabel_distr)
		classlabel_distr /= prob_sum

		#print("Final distribution over labels:\t"+str(classlabel_distr))
		return classlabel_distr[1]
	
	def classifyMultiplePredicates(self, object_id, behavior, predicates, trial):
		output_probs = []
		
		for p in predicates:
			output_probs.append(self.classify(object_id,behavior,p,trial))
		return output_probs
	
		
def main(argv):
		
		
	datapath = "../data/ijcai2016/corl_features"
	behaviors = ["look","grasp","lift","hold","lower","drop","push","press"]
	modalities = ["rgb","hsv","fc7","fpfh","color","surf","haptics","corl-audio"]
	#modalities = ["color","hsvnorm4","vgg","shape","effort","position","audio","surf200"]
	
	# minumum number of positive and negative examples needed for a predicate to be included in this experiment
	min_num_examples_per_class = 5 # in the actual experiment, this should be 4 or 5 to include more predicates; when doing a quick test, you can set it to 14 or 15 in which case only 3 predicates are valid

	# file that maps names to IDs
	objects_ids_file = "../data/ijcai2016/object_list.csv"
	
	# whether to load saved classifiers intead of training them
	# this should only be true if the procedure was first run once and the classifiers for all train-test splits were saved
	load_classifiers = False
	
	# some train parameters -- only valid if num_object_split_tests is not 32
	num_train_objects = 28
	num_trials_per_object = 5

	# whether to do internal cross validation; if false, all contexts are treated equally at test time
	# needs to be true if observation models for behaviors-predicate pairs are needed
	perform_internal_cv = True
	
	# how many total tests to do -- if 32, then this does object-based cross validation
	num_object_split_tests = 32
			
	# precompute and store train and test set ids for each test
	train_set_dict = dict()
	test_set_dict = dict()
	
	# create oracle
	T_oracle = TFTable(min_num_examples_per_class)
	T_oracle.loadFullAnnotations()
	
	# create classifier
	classifier = ClassifierIJCAI(datapath,behaviors,modalities,T_oracle,objects_ids_file)
	
	# get ids
	object_ids = copy.deepcopy(classifier.getObjectIDs());
	
	# set them in the oracle
	classifier._T_oracle.setObjectIDs(object_ids)
	
	# filter predicates
	print("All predicates:")
	all_predicates = T_oracle.getAllPredicates()
	print(str(all_predicates))
	print("Num predicates: "+str(len(all_predicates)))
	
	# where to store the confusion matrices for the classification results
	pred_cm_dict = dict()
	for pred in all_predicates:
		cm_p = np.zeros( (2,2) )
		pred_cm_dict[pred]=cm_p
	
	# compute train and test object sets for all tests
	if num_object_split_tests == 32: # doing leave one out object test
		object_ids = copy.deepcopy(classifier.getObjectIDs());
		
		for test in range(0,num_object_split_tests):
			test_objects = [object_ids[test]]
			train_objects = []
			for obj in object_ids:
				if obj != object_ids[test]:
					train_objects.append(obj)
			
			train_set_dict[test]=train_objects
			test_set_dict[test]=test_objects
	else:
		for test in range(0,num_object_split_tests):
			# get all object ids and shuffle them
			object_ids = copy.deepcopy(classifier.getObjectIDs());
			random.seed(test)
			random.shuffle(object_ids)
			train_object_ids = object_ids[0:num_train_objects]
			test_object_ids = object_ids[num_train_objects:len(object_ids)]
			train_set_dict[test]=train_object_ids
			test_set_dict[test]=test_object_ids
	
	# for each fold...
	for test in range(0,num_object_split_tests):
		
		print("\n"+"TEST "+str(test)+"\n")
		
		# create classifier
		classifier = ClassifierIJCAI(datapath,behaviors,modalities,T_oracle,objects_ids_file)
		
		# look up train and test objects for this test
		train_object_ids = train_set_dict[test]
		test_object_ids = test_set_dict[test]
		
		print("Train objects:\t"+str(train_object_ids))
		print("Test objects:\t"+str(test_object_ids))
		#print("size of train_object_ids: " + str(len(train_object_ids)))
		#print("size of object_ids: " + str(len(object_ids)))
		
		# figure out which predicates are required for evaluating on the test objects -- there is no need to train unneeded ones
		req_train_predicates = []
		for pred in all_predicates:
			for o in test_object_ids:
				if classifier._T_oracle.hasLabel(pred,str(o)):
					req_train_predicates.append(pred)
					break;
		print("# of required train predicates: "+str(len(req_train_predicates)))
		print(req_train_predicates)
		
		# where to save or load the classifier
		classifier_file_name = "pomdp_classifiers/classifier_test"+str(test)+".pkl"
			
		
		# train classifier
		if load_classifiers == False:
			classifier.trainClassifiers(train_object_ids,num_trials_per_object,req_train_predicates)
			
			# predicates that are known
			learned_predicates = classifier.learnedPredicates()
			print("Known predicates:\t"+str(learned_predicates))
			
			# predicates that occur in test object set
			test_predicates = []
			for i in range(0,len(learned_predicates)):
				for o in test_object_ids:
					if classifier._T_oracle.hasLabel(learned_predicates[i],str(o)):
						test_predicates.append(learned_predicates[i])
						break;		
		
			# perform cross validation to figure out context specific weights for each predicate (i.e., the robot should come up with a number for each sensorimotor context that encodes how good that context is for the predicate
			# this steps is needed to get the observation model for each behavior and predicate pair
		
			if perform_internal_cv:
				classifier.performCrossValidation(test_predicates)	
	
			# save classifier
			pkl_classifier_file = open(classifier_file_name, 'wb')
			pickle.dump(classifier, pkl_classifier_file)
			pkl_classifier_file.close()
		else:	
			# load classifier
			pkl_load_classifier_file = open(classifier_file_name, 'rb')
			classifier = pickle.load(pkl_load_classifier_file)
			
			# predicates that are known
			learned_predicates = classifier.learnedPredicates()
			print("Known predicates:\t"+str(learned_predicates))
			
			# predicates that occur in test object set
			test_predicates = []
			for i in range(0,len(learned_predicates)):
				for o in test_object_ids:
					if classifier._T_oracle.hasLabel(learned_predicates[i],str(o)):
						test_predicates.append(learned_predicates[i])
						break;
		
		# test on remaining objects
		print("Test objects:")
		print(test_object_ids)
	
		for o_test in test_object_ids:
			print("test object:\t"+str(o_test)+"\t"+classifier.getObjectName(o_test))
			
			# for all predicates, if a label is available for this test object, then perform classification and record result
			for p in range(0,len(test_predicates)):
				pred = test_predicates[p]
				if classifier._T_oracle.hasLabel(pred,str(o_test)):
					print("\t\tpredicate: "+pred+":")
					for t in range(1,num_trials_per_object+1):
						
						#probs_b = classifier.classifyMultiplePredicates(o_test,behaviors,[pred],t)
						#print(probs_b)
						
						# get classifier output
						c_output = classifier.classify(o_test,behaviors,pred,t)
						
						# compute actual and predicted integers
						actual = 1 if classifier.isPredicateTrue(pred,o_test) else 0
						predicted = 1 if c_output > 0.5 else 0		
						
						# update confusion matrix				
						pred_cm_dict[pred][predicted][actual] = pred_cm_dict[pred][predicted][actual] + 1
			
			
			# example of how to get an observation model for a given predicate and behavior pair
			# note that the 2 x 2 confusion matrix is already normalized so that its entries sum up to 1.0
			CM_pb = classifier.getPredicateBehaviorObservatoinModel("water","look")
			print("\nObservation model for sample predicate and behavior:")
			print(CM_pb)
			
			
			# example of how to get prob. estimates for a set of a predicates, an object, and a behavior:
			query_predicates = ["bright", "medium-sized", "light"]
			query_behavior_in_list = ["look"]
			query_trial_index = 1 # 1-5, as each behavior was executed 5 times on each object
			
			query_pred_probs = classifier.classifyMultiplePredicates(o_test,query_behavior_in_list,query_predicates,query_trial_index)
			print("\nExample query predicates and probabilities:")
			print(query_predicates)
			print(query_pred_probs)
			
		
			
	# print final statistics about recognition performance when all behaviors are applied on each test object		
	print("\n\nFinal kappas:\n")
	for pred in all_predicates:
		cm_p = pred_cm_dict[pred]
		#print(cm_p)
		if np.sum(cm_p) > 0:
			print(pred+","+str(classifier.computeKappa(cm_p)))
		
		
	
if __name__ == "__main__":
    main(sys.argv)
