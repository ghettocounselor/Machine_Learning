#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 05:08:44 2019

@author: markloessi
"""

# from here: https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176
# another one: https://chrisalbon.com/machine_learning/trees_and_forests/visualize_a_decision_tree/

# import basic data
import sklearn.datasets as datasets
import pandas as pd
iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target

# generate decision tree and fit to data
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)

# make a visualization
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_pdf("decision_tree_vis.pdf")
graph.write_png("decision_tree_vis.png")