import pandas as pd
import numpy as np

import csv	

import pprint
import pickle

'''The TFTable (true false table) contains two functions,
 the getTorF function returns true/false/none based on given predicate and objectID
 the getAllPredicates function return all predicates as a list (105 int total)'''

class TFTable:
    def __init__(self,min_num_examples_per_class):
        table_path = "../data/ijcai2016/labels.csv"
        self.df = pd.read_csv(table_path,index_col=0)
        self.missing_as_negative = False	
        self.obj_ids = []
        self.annotations = None
        self._min_num_examples_per_class = min_num_examples_per_class
        #self.predicates_annotated = pd.read_csv("../data/ijcai2016/test_full.csv",index_col=0)
        
        #print(self.predicates_annotated)
        
        
        
        with open('../data/ijcai2016/test_full.csv', 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                self._predicates_annotated = row
        
    def loadFullAnnotations(self):
        # load full annotations
        pkl_file = open('../data/ijcai2016/full_annotations_ijcai2016.pickle', 'rb')
		
        data1 = pickle.load(pkl_file)
        print("Loaded pickle.")
        
        predicates = self._predicates_annotated
        #print(predicates)
        
        self.class_label_dict = dict()
        for file_obj_id in range(0,32):
			# this is waht needs to be stored as because I use 1-32, not 0-31
            valid_object_id = file_obj_id + 1
			
            # for each predicate, we store output here
            obj_label_dict = dict()
            
            # get the labels in a list form from the pickle
            obj_labels = data1[file_obj_id]
            
            #print(len(predicates))
            
            # for each predicate, store in dict
            for p in range(0,len(predicates)):
                #print(p)
                #print(len(obj_labels))
                label = obj_labels[p]
                obj_label_dict[predicates[p]] = label
				
            self.class_label_dict[valid_object_id]=obj_label_dict
        #print(self.class_label_dict[3])
       
		
    def setObjectIDs(self,obj_ids):
        self.obj_ids = obj_ids	
		
    def getTorF(self,predicate, objectID):
        self.predicate = predicate
        self.objectID = objectID
        #ret = self.df.ix[predicate,objectID]
        ret = self.class_label_dict[int(objectID)][predicate]
        
        if (ret==1):
            return True
        else:
            return False

    def hasLabel(self,predicate,objectID):
		
        return True
    
    def getAllPredicates(self):
		
		# all
        pred_candidates = self._predicates_annotated
		
		# color based
        #pred_candidates = ["yellow","green","blue","black","brown","bright","colored","color","cream-colored","gray","metal","metallic","neon","orange","pink","purple","red","silver","shiny","white"]

		
        self.pred = pred_candidates
        
		# for each predicate see if min number of examples is met
        min_num_positive = self._min_num_examples_per_class
        min_num_negative = self._min_num_examples_per_class
		
        if len(self.obj_ids) == 0:
            return self.pred
        else:
            pred_counts_pos = np.zeros(len(self.pred))
            pred_counts_neg = np.zeros(len(self.pred))
            for p in range(0,len(self.pred)):
                 for o in self.obj_ids:
                      if self.hasLabel(self.pred[p],str(o)):
                           if self.getTorF(self.pred[p],str(o)):
                                pred_counts_pos[p] = pred_counts_pos[p] + 1       
                           else:
                                pred_counts_neg[p] = pred_counts_neg[p] + 1  
        
        pred_filtered = []
        
        for p in range(0,len(self.pred)):
            if pred_counts_pos[p] > min_num_positive and pred_counts_neg[p] > min_num_negative:
                pred_filtered.append(self.pred[p])
				
        
        return pred_filtered
