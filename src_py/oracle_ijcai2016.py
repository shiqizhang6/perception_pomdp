import pandas as pd
import numpy as np
'''The TFTable (true false table) contains two functions,
 the getTorF function returns true/false/none based on given predicate and objectID
 the getAllPredicates function return all predicates as a list (105 int total)'''

class TFTable:
    def __init__(self):
        table_path = "../data/labels.csv"
        self.df = pd.read_csv(table_path,index_col=0)

    def getTorF(self,predicate, objectID):
        self.predicate = predicate
        self.objectID = objectID
        ret = self.df.ix[self.predicate,self.objectID]
       # print ret
        if (ret==1.0):
            return True
        elif (ret == -1.0):
            return False
        else:
            return None

    def getAllPredicates(self):
        self.pred = self.df.index.tolist()
        return self.pred

# t=TFTable()
# pred = t.getAllPredicates()
# for id in np.arange(1,33):
#     for p in pred:
#         print p,id,t.getTorF(p,str(id))
