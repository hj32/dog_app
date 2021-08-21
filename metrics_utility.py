
from keras import backend as K

### Define F1 measures: F1 = 2 * (precision * recall) / (precision + recall)

def f1_score(y_true, y_pred):    
    """
    Calculate F1 Score 
    Args:
    y_true (tensor): targets actual
    y_pred (tensor): targets predicted
    Returns:
    F1 Score calculated
    """
    def recall_m(y_true, y_pred):
        """
        Calculate Recall
         Args:
         y_true (tensor): targets actual
         y_pred (tensor): targets predicted
         Returns:
         recall calculated
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives + K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        """
        Calculate precision
         Args:
         y_true (tensor): targets actual
         y_pred (tensor): targets predicted
         Returns:
         Precision calculated
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives + K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2 * ((precision*recall) / (precision+recall+K.epsilon()))