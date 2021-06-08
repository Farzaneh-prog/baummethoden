from sklearn.tree import export_graphviz
import sys, os
sys.path.append("..")

def Baum_tree(classifier, attribute_names):
    
    #change the working directory
    path_start = os.getcwd()
    pathr=os.path.dirname(os.getcwd())+'/reports/figures'
#    pathr=os.path.dirname(os.getcwd())+'/../reports/figures'

    os.chdir(pathr)
    export_graphviz(classifier, out_file=("tree_bills.dot"), feature_names=attribute_names[0:4],class_names=(['real','fake']), rounded=True, filled=True)
    os.system("dot -Tpng tree_bills.dot -o tree_bills.png") 
    os.system("dot -Tps tree_bills.dot -o tree_bills.ps")

    #Export dot to png 
    #from subprocess import check_call
    #check_call(['dot','-Tpng','tree_bills.dot','-o','tree_bills.png'])
    
    #change to the start working directory
    os.chdir(path_start)
    return