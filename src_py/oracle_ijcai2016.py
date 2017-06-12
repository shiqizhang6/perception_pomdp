import pandas as pd
import numpy as np
'''The TFTable (true false table) contains two functions,
 the getTorF function returns true/false/none based on given predicate and objectID
 the getAllPredicates function return all predicates as a list (105 int total)'''

class TFTable:
    def __init__(self):
        table_path = "../data/ijcai2016/labels.csv"
        self.df = pd.read_csv(table_path,index_col=0)
        self.missing_as_negative = False	
        self.obj_ids = []
		
    def setObjectIDs(self,obj_ids):
        self.obj_ids = obj_ids	
		
    def getTorF(self,predicate, objectID):
        self.predicate = predicate
        self.objectID = objectID
        ret = self.df.ix[predicate,objectID]
        
        
        
       # print ret
        if (ret==1.0):
            return True
        elif (ret == -1.0):
            return False
        else:
            return False

    def hasLabel(self,predicate,objectID):
		
        if self.missing_as_negative:
            return True
		
        #print(predicate)
        #print(objectID)
        self.predicate = predicate
        self.objectID = objectID
        
        ret = self.df.ix[self.predicate,self.objectID]
        if (ret==1.0):
            return True
        elif (ret == -1.0):
            return True
        else:
            return False
    
    def getAllPredicates(self):
		
		# all
        pred_candidates = self.df.index.tolist()
		
		# color based
        #pred_candidates = ["yellow","green","blue","black","brown","bright","colored","color","cream-colored","gray","metal","metallic","neon","orange","pink","purple","red","silver","shiny","white"]

		
        self.pred = pred_candidates
        
		# for each predicate see if min number of examples is met
        min_num_positive = 3
        min_num_negative = 3
		
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
        
        #print(pred_counts_pos)
        #print(pred_counts_neg)
        
        pred_filtered = []
        
        for p in range(0,len(self.pred)):
            if pred_counts_pos[p] > min_num_positive and pred_counts_neg[p] > min_num_negative:
                pred_filtered.append(self.pred[p])
				
        
        return pred_filtered
# t=TFTable()
# pred = t.getAllPredicates()
# for id in np.arange(1,33):
#     for p in pred:
#         print p,id,t.getTorF(p,str(id))
