class TruthCounter:
    """
    Class to store Truth table of results from decisions made by each algorithm
    """
    
    def __init__(self):
        self.__true_positive__=0
        self.__true_negative__=0
        self.__false_positive__=0
        self.__false_negative__=0
        
    
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
        """
        print('{:<4}{:<7}{:<9}{:<4}'.format('TP\t','TN\t','FP\t','FN\t'))
        print('{:<10}{:<10}{:<10}{:<10}'.format(self.__true_positive__,
        self.__true_negative__,
        self.__false_positive__,
        self.__false_negative__))
    

class Scoreboard:
    
    
    
    def __init__(self):
        self.__dd_counter__=TruthCounter()# dog detector counter
        self.__hd_counter__=TruthCounter()# human detector counter
        self.__db_counter__=TruthCounter()
        
    def set_dog_detector_score(self,  isDogActual=False, isDogPred=False):
        """
        
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
         if( isHumanActual and isHumanPred):
            self.__hd_counter__.add_true_positive()
         elif(isHumanActual and not isHumanPred):
            self.__hd_counter__.add_false_negative()
         elif(not isHumanActual and isHumanPred):
            self.__hd_counter__.add_false_positive()
         elif(not isHumanActual and not isHumanPred):
            self.__hd_counter__.add_true_negative()

    def set_dog_breed_detector_score(self, breedPred="", breedActual=""):
        if( breedPred == breedActual):
            self.__db_counter__.add_true_positive()
        elif(breedPred != breedActual):
            self.__db_counter__.add_false_negative()
        
    def print_out(self):
        print("Dog Detector")
        self.__dd_counter__.print_out()
        print("Human Detector")
        self.__hd_counter__.print_out()
        print("Dog Breed Detector")
        self.__db_counter__.print_out()