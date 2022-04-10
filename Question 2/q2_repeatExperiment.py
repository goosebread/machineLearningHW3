from q2_GMM_fit import runTrainingSet
import pandas as pd
import numpy as np


#do experiment
#run script
if __name__ == '__main__':
    printNLLs = False #use this to verify that the repetitions actually have different results

    train_sets = ["Samples10","Samples100","Samples1000","Samples10000"]
    selections = pd.DataFrame(columns=train_sets)
    repetitions = 30
    for i in range(repetitions):
        print(i)
        selectionRow = np.zeros(len(train_sets))
        for s in range(len(train_sets)):
            print(train_sets[s])
            [selectedModel,averageNLLS] = runTrainingSet(train_sets[s],True,True)
            selectionRow[s] = selectedModel
            if printNLLs:
                print(averageNLLS)
        selections.loc[i] = selectionRow

    #output results
    selections.to_csv("Q2_Repetition_Results.csv")
    print(selections)


