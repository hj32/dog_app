from numpy import False_


class TruthCounter:
    
    
    def __init__(self,tp=0,tn=0,fp=0,fn=0):
        """
            Args:
            tp(int): True Positive
            tn(int): True Negative
            fp(int): False Positive
            fn(int): False Negative
        """
        self.__true_positive__=tp
        self.__true_negative__=tn
        self.__false_positive__=fp
        self.__false_negative__=fn
        
    
    def add_true_positive(self):
        """
        Increment true positive count
        """
        self.__true_positive__+=1
    def add_true_negative(self):
        """
        Increment true negative count
        """
        self.__true_negative__+=1
    def add_false_positive(self):
        """
        Increment false positive count
        """
        self.__false_positive__+=1
    def add_false_negative(self):
        """
        Increment false negative count
        """
        self.__false_negative__+=1
    
        
    def print_out(self):
        """
        Print out counts
        Returns :
        zero : Zero better than None
        """
        print('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('TP','TN','FP','FN','ChkSum','Acc'))
        print('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(self.__true_positive__,
        self.__true_negative__,
        self.__false_positive__,
        self.__false_negative__,
        self.get_check_sum(),
        np.round(self.get_accuracy(self.__true_positive__,self.__true_negative__,
        self.__false_positive__,self.__false_negative__),4)))
        return 0


    def get_check_sum(self):
        """
        Return sum of all 4 counts
        """
        return self.__true_positive__ + self.__true_negative__+ self.__false_positive__ + self.__false_negative__
    
    def get_accuracy(self,true_positive,true_negative,false_positive,false_negative):
        """
        Args:
        true_positive (int):  Count True Positives
        true_negative (int):  Count True Negatives
        false_positive (int):  Count False Positives
        false_negative (int):  Count False Negatives
        Returns:
        Accuracy (float)
        """
        if(true_positive + true_negative+ false_positive + false_negative>0):
            accuracy=(true_positive + true_negative) / (true_positive + true_negative+ false_positive + false_negative)
        else:
            accuracy=-1
        return accuracy

    

class Scoreboard:
    
    
    
    def __init__(self):
        self.__dd_counter__=TruthCounter()# dog detector counter
        self.__hd_counter__=TruthCounter()# human detector counter
        self.__db_counter__=TruthCounter()# dog breed counter
        
    def set_dog_detector_score(self,  isDogActual=False, isDogPred=False):
        """
        Set dog detector score
        """
        if( isDogActual and isDogPred):
            self.__dd_counter__.add_true_positive()
        elif(isDogActual and not isDogPred):
            self.__dd_counter__.add_false_negative()
        elif(not isDogActual and isDogPred):
            self.__dd_counter__.add_false_positive()
        elif(not isDogActual and not isDogPred):
            self.__dd_counter__.add_true_negative()
            
    def set_human_detector_score(self,  isHumanActual=False, isHumanPred=False,):
         """
         set human detector score
         """
         if( isHumanActual and isHumanPred):
            self.__hd_counter__.add_true_positive()
         elif(isHumanActual and not isHumanPred):
            self.__hd_counter__.add_false_negative()
         elif(not isHumanActual and isHumanPred):
            self.__hd_counter__.add_false_positive()
         elif(not isHumanActual and not isHumanPred):
            self.__hd_counter__.add_true_negative()

    def set_dog_breed_detector_score(self, breedPred="", breedActual=""):
        """
        Set breed detector score
        """
        if( breedPred == breedActual):
            self.__db_counter__.add_true_positive()
        elif(breedPred != breedActual):
            self.__db_counter__.add_false_positive()
    
        
    def print_out(self):
        print("Dog Detector")
        self.__dd_counter__.print_out()
        print("Human Detector")
        self.__hd_counter__.print_out()
        print("Dog Breed Detector")
        self.__db_counter__.print_out()



import numpy as np



