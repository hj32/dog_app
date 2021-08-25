import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class AccuracyScorer:
    """
    Class to score accuracy of models
    """

    def __init__(self):
        self.__actual__=[]
        self.__preds__=[]

        self.__true_positive__=[]
        self.__true_negative__=[]
        self.__false_positive__=[]
        self.__false_negative__=[]

    def add_result(self,actual_value, pred_value):
        """
        Append results to accual and predicted lists
        Args:
        actual_value (int): actual value to add
        pred_value (int): pred value to add
        """
        self.__actual__.append(actual_value)
        self.__preds__.append(pred_value)


    def add_up(self,arg_list):
        """
        Extend add method to cope with more than 2 arguments
        Args:
        arg_list (list): List of arguments

        Returns:
        result : consecutive  np.Add() operations add all the arguments together
        """
        if(len(arg_list)>1):
            result=arg_list[0]
            for i in range(1, len(arg_list)):
               result = np.add(result, arg_list[i])
        return result


    def get_accuracy(self):
        """
        Calculate accuracy
        Returns:
        Accuracy_Score (float):Accuracy score
        """
        cnf_matrix = confusion_matrix(self.__actual__, self.__preds__)

        self.__false_positive__= np.subtract(cnf_matrix.sum(axis=0), np.diag(cnf_matrix))
        self.__false_negative__= np.subtract(cnf_matrix.sum(axis=1), np.diag(cnf_matrix))
        self.__true_positive__ = np.diag(cnf_matrix)
        self.__true_negative__ = np.subtract(cnf_matrix.sum() , self.add_up([self.__false_positive__, self.__false_negative__, self.__true_positive__]))

        return accuracy_score(self.__actual__, self.__preds__)

    def print_truth(self):
        """
        Print the True Positives False Positives, True Negatives, False Negatives for the confusion matrix of the actual and predictions
        """
        print("TP",self.__true_positive__)
        print("TN",self.__true_negative__)
        print("FP",self.__false_positive__)
        print("FN",self.__false_negative__)










