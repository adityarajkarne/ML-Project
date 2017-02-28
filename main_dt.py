
from functions_dt import max_information_gain, build_tree, print_tree, test_row, print_fulltree
from Node import Node
import pandas as pd

# Global Variables
diabetes_accuracy = 0
monk_accuracy = 0
diabetes_confusion_matrix = {}
monk_confusion_matrix = {}
depth_list = []
accuracy_list = []
accuracy_wrt_file = {}
training_datasets = ['diabetes_train.csv', 'monks-1.train.txt', 'monks-2.train.txt', 'monks-3.train.txt']
test_datasets = ['diabetes_test.csv', 'monks-1.test.txt', 'monks-2.test.txt', 'monks-3.test.txt']

# start = tm.clock()


# Print confusion matrix
def print_confusion_matrix(confusion_matrix, nb_type):
    # print ("\nConfusion Matrix")
    # print (confusion_matrix)
    print("\n" + nb_type + " Confusion Matrix")
    print("\t Model Results")
    print("  \t T \t    N \tTotal")
    print("T \t" + str(confusion_matrix['TP']) + "    " + str(confusion_matrix['FN']) + "\t" + str(confusion_matrix['AP']))
    print("N \t" + str(confusion_matrix['FP']) + "    " + str(confusion_matrix['TN']) + "\t" + str(confusion_matrix['AN']))
    print("\n")


# main function for decision tree
def decision_tree():
    print('************************************Decision Tree****************************************')
    print('')
    # Training
    # loop for each set of training and testing files
    for j in range(0, len(training_datasets)):
        training_dataset_file = training_datasets[j]
        test_dataset_file = test_datasets[j]
        depth = 0
        df = pd.read_csv(training_dataset_file, sep=",")
        # storing data in Pandas Dataframe
        train_dataset = pd.DataFrame(df)

        # get the initial best column name and vaue for root node
        best_column, best_column_value = max_information_gain(train_dataset)
        root_node = Node(train_dataset, depth, best_column, best_column_value)
        depth_threshold = 20
        # build tree recursively
        build_tree(root_node, depth_threshold)

        # Testing
        df = pd.read_csv(test_dataset_file, sep=",")
        test_dataset = pd.DataFrame(df)
        test_dataset_count = test_dataset.shape[0]
        column_names = list(test_dataset.columns)
        class_column = column_names[0]

        # calculating depth 0 accuracy for each file
        class_0_count = test_dataset[test_dataset[class_column] == 0].shape[0]
        class_1_count = test_dataset[test_dataset[class_column] == 1].shape[0]

        # initialize TP, FN, TN and FP to 0
        TP = 0
        FN = 0

        TN = 0
        FP = 0

        predict_right_count = 0
        for index, row in test_dataset.iterrows():
            predicted_class = test_row(root_node, row)
            if predicted_class == row[class_column]:
                predict_right_count += 1
                if predicted_class == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if predicted_class == 1:
                    FN += 1
                else:
                    FP += 1

        if 'diabetes' in test_dataset_file:
            diabetes_accuracy = (100.0 * predict_right_count)/test_dataset_count
            diabetes_confusion_matrix['AP'] = class_1_count
            diabetes_confusion_matrix['AN'] = class_0_count
            diabetes_confusion_matrix['TP'] = TP
            diabetes_confusion_matrix['FN'] = FN
            diabetes_confusion_matrix['TN'] = TN
            diabetes_confusion_matrix['FP'] = FP

            print('\nDiabetes Results:')
            print('Diabetes Accuracy:', diabetes_accuracy)
            print('Confusion Matrix:')
            print_confusion_matrix(diabetes_confusion_matrix, "Diabetes")
        else:
            monk_accuracy = (100.0 * predict_right_count)/test_dataset_count
            monk_confusion_matrix['AP'] = class_1_count
            monk_confusion_matrix['AN'] = class_0_count
            monk_confusion_matrix['TP'] = TP
            monk_confusion_matrix['FN'] = FN
            monk_confusion_matrix['TN'] = TN
            monk_confusion_matrix['FP'] = FP

            print(test_dataset_file, ' Results')
            print('Accuracy:', monk_accuracy)
            print('Confusion Matrix:')
            print_confusion_matrix(monk_confusion_matrix, "Monk")


