from torch import logical_and as t_and, sum as t_sum

def table(truths, predictions, threshold = .5, labels = [1]):
    predictions = predictions >= threshold
    
    tp = t_sum(t_and(predictions == True,  truths == True )[:,labels])
    tn = t_sum(t_and(predictions == False, truths == False)[:,labels])
    fp = t_sum(t_and(predictions == True,  truths == False)[:,labels])
    fn = t_sum(t_and(predictions == False, truths == True )[:,labels])

    return tp, tn, fp, fn

def stats(stats):
    tp, tn, fp, fn = stats

    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    recall    = (tp) / (tp + fn) # = 1 - fn/(tp+fn)
    precision = (tp) / (tp + fp) # = 1 - fp/(tp+fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, recall, precision, f1