#! /usr/bin/env python


import random
import os
import numpy as np
import sys
import csv
import copy

# classifiers
from sklearn import svm
from sklearn.preprocessing import normalize

class ClassifierICRA(object):
	
	def __init__(self, data_path, behaviors, modalities, predicates):
		# load data
		self._path = data_path
		self._behaviors = behaviors
		self._modalities = modalities
		self._predicates = predicates
		
		self._num_interaction_trials = 0
		
		# some constants
		self._num_trials_per_object = 10
		self._train_test_split_fraction = 2/3 # what percentage of data is used for training when doing internal cross validation on training data
		
		# stores training data
		self._predicate_obj_data_dict = dict()
		
		# compute lists of contexts
		self._contexts = []
		
		for b in behaviors:
			for m in modalities:
				if self.isValidContext(b,m):
					context_bm = b+"-"+m
					self._contexts.append(context_bm)
		
		# print to verify
		#print("Valid contexts:")
		#print(self._contexts)
		
		# dictionary that holds context specific weights for each predicate
		self._pred_context_weights_dict = dict()
		
		# load object ids
		self._object_ids = []
		object_file = self._path +"/objects.txt"
		with open(object_file, 'r') as f:
			reader = csv.reader(f)
			for row in reader:
				self._object_ids.append(row[0])	
				
		#print("Set of objects:")
		#print(self._object_ids)
		
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
			
			#print('Loading ' + context_filename+ '...')
			with open(context_filename, 'r') as f:
				reader = csv.reader(f)
				for row in reader:
					obj = row[0]
					features = row[1:len(row)]
					object_trial_count_dict[obj] += 1
					
					key = obj+"_"+str(object_trial_count_dict[obj])
					data_dict[key] = features
					
			
			self._context_db_dict[context] = data_dict
	
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
	
	def getFeatures(self,context,object_id,trial_number):
		key = object_id+"_"+str(trial_number)
		return self._context_db_dict[context][key]
	
	def getObjectIDs(self):
		return self._object_ids
	
	def isPredicateTrue(self,predicate,object_id):
		if predicate in object_id:
			return True
		return False
	
	# inputs: learn_prob_model (either True or False)
	def createScikitClassifier(self, learn_prob_model):
		# currently an SVM
		return svm.SVC(gamma=0.001, C=100, probability = learn_prob_model)
	
	def crossValidateObjectBased(self, predicate, context):
		
		#print("cross-validating "+context+" classifier for "+predicate)
		
		data_dict_cp = self._predicate_obj_data_dict[predicate][context]
		objects_cp = data_dict_cp.keys()
		
		# confusion matrix
		CM_total = np.zeros( (2,2) )
		
		for test_object in objects_cp:
			train_objects = []
			for o in objects_cp:
				if o != test_object:
					train_objects.append(o)
			
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
	
		score = self.computeKappa(CM_total)
		return score
		#print(CM_total)
	
	def crossValidate(self,X,Y,num_tests):
		scores = []
		
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
			
			X_f_test = X_f[int(len(X_f)*2/3):int(len(X_f))]
			Y_f_test = Y_f[int(len(Y_f)*2/3):int(len(Y_f))]
			
			# create and train classifier
			classifier_f = self.createScikitClassifier(False)
			classifier_f.fit(X_f_train, Y_f_train)
			
			
			score_f = classifier_f.score(X_f_test, Y_f_test) 
			scores.append(score_f)
		mean_score = np.mean(scores)
		#print mean_score
		return mean_score
		  
	
	def performCrossValidation(self, num_tests):
		for predicate in self._predicates:
			print("Cross-validating classifiers for "+predicate)
			# this contains the context-specific classifier for the predicate
			
			# check if classifier for predicate exists -- it may be possible that the set of training objects only contain positive or only contain negative examples in which case the classifier wouldn't have been created
			if predicate in self._predicate_classifier_dict.keys():
			
				classifier_ensemble_dict = self._predicate_classifier_dict[predicate]
			
				# this contains the data for the predicates
				pred_data_dict = self._predicate_data_dict[predicate]
			
				pred_context_weights = dict()
				for context in self._contexts:
					# object based cv
					
					score_cp = self.crossValidateObjectBased(predicate, context)
					
					if score_cp < 0.01:
						score_cp = 0.01
					#else:
					#	print("Score for " + predicate + " and context "+context +"\t"+str(score_cp))
					
					#[X,Y] = pred_data_dict[context]
					#print("Cross-validating predicate " + predicate + " and context "+context+" with " + str(len(X)) + " points")
					#score_cp = self.crossValidate(X,Y,num_tests)
				
					# store the weight for that context and predicate
					pred_context_weights[context]=score_cp
					#pred_context_weights[context]=1.0
				
				self._pred_context_weights_dict[predicate] = pred_context_weights
			else:
				print("[WARN] Cannot perform CV for predicate "+predicate)
		#print "Context weight dict:"
		#print self._pred_context_weights_dict
	
	# train_objects: a list of training objects
	# num_interaction_trials:	how many datapoint per object, from 1 to 10
	def trainClassifiers(self,train_objects,num_interaction_trials):
		
		self._num_interaction_trials = num_interaction_trials
		
		# for each predicate
		
		# dictionary storing the ensemble of classifiers (one per context) for each predicate
		self._predicate_classifier_dict = dict()
		self._predicate_data_dict = dict()
		
		# dictionary that stores predicate, context, object data dictionary
		self._predicate_obj_data_dict = dict()
		
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
			
			#print("Positive examples: "+str(positive_object_examples))
			#print("Negative examples: "+str(negative_object_examples))
			
			# this dictionary stores obj_data_dict
			obj_pred_data_dict = dict()
			
			# train classifier for each context
			for context in self._contexts:
				# create dictionary with object ids
				obj_data_dict = dict()
				
				all_objects = positive_object_examples + negative_object_examples
				
				for o in all_objects:
					obj_data_dict[o] = []
				
				# create dataset for this context 
				X = []
				Y = []
				
				for o in positive_object_examples:
	
					for t in range(1,num_interaction_trials+1):
						x_ot = self.getFeatures(context,o,t)
						y_ot = 1
						X.append(x_ot)
						Y.append(y_ot)
						
						obj_data_dict[o].append([x_ot,y_ot])
						
				for o in negative_object_examples:
					for t in range(1,num_interaction_trials+1):
						x_ot = self.getFeatures(context,o,t)
						y_ot = 0
						X.append(x_ot)
						Y.append(y_ot)
						
						obj_data_dict[o].append([x_ot,y_ot])
				
				#print(str(obj_data_dict))
				
				# store object-based dataset
				obj_pred_data_dict[context] = obj_data_dict
				
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
			
			# store data in data dictionary
			self._predicate_obj_data_dict[predicate] = obj_pred_data_dict
			
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
	
	def getBehaviorScore(self,behavior,predicate):
		
		max_context_score = 0.0
		for context in self._contexts:
			if behavior in context:
				context_weight = self._pred_context_weights_dict[predicate][context]
				if context_weight > max_context_score:
					max_context_score = context_weight
		return max_context_score
	
	def classifyMultipleBehaviors(self, object_id, behaviors, predicate, trial_index):
		# before doing anything, check whether we even have classifiers for the predicate
		if predicate not in self._predicate_classifier_dict.keys():
			# return negative result
			print("[WARN] predicate "+predicate+" is not known...")
			return 0.0
			
		# first, randomly pick which trial we're doing
		selected_trial = trial_index
		
		# next, find which contexts are available in that behavior
		b_contexts = []
		for context in self._contexts:
			for behavior in behaviors:
				if behavior in context:
					b_contexts.append(context)
		# output distribution over class labels (-1 and +1)
		# print(b_contexts)
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
				
			#print(context+"\t"+str(context_weight))
			
			classlabel_distr += context_weight*output[0]
			
			#if context_weight > 0:
			#	print("\tPrediction from context "+context+":\t"+str(output))
		
		# normalize so that output distribution sums up to 1.0
		prob_sum = sum(classlabel_distr)
		classlabel_distr /= prob_sum

		#print("Final distribution over labels:\t"+str(classlabel_distr))
		return classlabel_distr[1]	
			
	# input: the target object, the behavior, and a predicate
	# output: the probability that the object matches the predicate		
	def classify(self, object_id, behavior, predicate):
		
		# before doing anything, check whether we even have classifiers for the predicate
		if predicate not in self._predicate_classifier_dict.keys():
			# return negative result
			return 0.0
			
			
		# first, randomly pick which trial we're doing
		num_available = self._num_trials_per_object
		selected_trial = random.randint(1,num_available)
		
		# next, find which contexts are available in that behavior
		b_contexts = []
		for context in self._contexts:
			if behavior in context:
				b_contexts.append(context)
		
		#print b_contexts
		#print selected_trial
		
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
				
			#print context_weight
			
			classlabel_distr += context_weight*output[0]
			#print("Prediction from context "+context+":\t"+str(output))
		
		# normalize so that output distribution sums up to 1.0
		prob_sum = sum(classlabel_distr)
		classlabel_distr /= prob_sum

		#print("Final distribution over labels:\t"+str(classlabel_distr))
		return classlabel_distr[1]
	
	def classifyMultiplePredicates(self, object_id, behavior, predicates):
		output_probs = []
		
		for p in predicates:
			output_probs.append(self.classify(object_id,behavior,p))
		return output_probs
	
		
def main(argv):
		
	datapath = "../data/icra2014"
	behaviors = ["look","grasp","lift_slow","hold","shake","high_velocity_shake","low_drop","tap","push","poke","crush"]
	modalities = ["color","patch","proprioception","audio"]

	predicates = ['brown','green','blue','light','medium','heavy','glass','screws','beans','rice']


	classifier = ClassifierICRA(datapath,behaviors,modalities,predicates)
	print("Classifier created...")
	
	# some train parameters
	num_train_objects = 24
	num_trials_per_object = 10
	
	# how train-test splits to use when doing internal cross-validation (i.e., cross-validation on train dataset)
	num_cross_validation_tests = 5
	
	
	# get all object ids and shuffle them
	object_ids = copy.deepcopy(classifier.getObjectIDs());
	
	random.seed(1)
	random.shuffle(object_ids)
	#print object_ids

	
	# do it again to check that the random seed shuffles the same way
	#object_ids2 = classifier.getObjectIDs();
	#random.seed(1)
	#random.shuffle(object_ids2)
	#print object_ids2
	
	
	# pick random subset for train
	train_object_ids = object_ids[0:num_train_objects]
	#print train_object_ids
	print("size of train_object_ids: " + str(len(train_object_ids)))
	print("size of object_ids: " + str(len(object_ids)))
	
	# train classifier
	classifier.trainClassifiers(train_object_ids,num_trials_per_object)
	
	# perform cross validation to figure out context specific weights for each predicate (i.e., the robot should come up with a number for each sensorimotor context that encodes how good that context is for the predicate
	classifier.performCrossValidation(5)
	
	# optional: reset random seed to something specific to this evaluation run (after cross-validation it is fixed)
	random.seed(235)
	
	# test classifying an object based on a single behavior and 1 predicate
	target_object = object_ids[num_train_objects+1]
	behavior = "look"
	query_predicate = "blue"
	
	print("\nTarget object: "+target_object+"\nbehavior: "+behavior+"\npredicate: "+query_predicate)
	
	output_prob = classifier.classify(target_object,behavior,query_predicate)
	
	print("Predicate probability score:\t"+str(output_prob))
	
	# test classifying multiple predicates using a single behavior
	query_predicate_list = ['light','medium','heavy']
	print("\nPredicate list query:\t"+str(query_predicate_list))
	
	output_probs = classifier.classifyMultiplePredicates(target_object,behavior,query_predicate_list)
	print("Output probs.:\t"+str(output_probs))
	
	
if __name__ == "__main__":
    main(sys.argv)
