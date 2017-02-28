import pandas as pd
import numpy as np


# function to create new centroids
def create_new_centroids(train_dataset, centroids, sse):
    columns = list(train_dataset.columns)
    rows_in_centroid_1 = pd.DataFrame(columns=columns)
    rows_in_centroid_2 = pd.DataFrame(columns=columns)
    new_sse = 0

    for index, row in train_dataset.iterrows():
        distance_wrt_centroid = {}
        for centroid_index in centroids.keys():
            distance = 0
            for i in range(1, len(columns)):
                column = columns[i]
                # Euclidian Distance
                distance += np.square(centroids[centroid_index][column] - row[column])
            distance = format(np.sqrt(distance), '.2f')
            distance_wrt_centroid[centroid_index] = distance
        centroid_index = min(distance_wrt_centroid, key=distance_wrt_centroid.get)
        min_dist = float(min(list(distance_wrt_centroid.values())))
        # Calculating Sum of Squared errors and summing it for all clusters
        new_sse += min_dist*min_dist

        if centroid_index == 1:
            rows_in_centroid_1.loc[index] = row
        else:
            rows_in_centroid_2.loc[index] = row

    new_centroids = centroids

    no_of_rows_in_centroid1 = rows_in_centroid_1.shape[0]
    no_of_rows_in_centroid2 = rows_in_centroid_2.shape[0]
    # Updating old centroid values with new values
    for i in range(1, len(columns)):
        column = columns[i]
        avg_value1 = float(sum(list(rows_in_centroid_1[column]))/no_of_rows_in_centroid1)
        new_centroids[1][column] = avg_value1
        avg_value2 = float(sum(list(rows_in_centroid_2[column]))/no_of_rows_in_centroid2)
        new_centroids[2][column] = avg_value2

    # Calculating class for each centroid
    centroid_1_class = int(rows_in_centroid_1['class'].mode())
    centroid_2_class = int(rows_in_centroid_2['class'].mode())

    centroid_1_count = rows_in_centroid_1.shape[0]
    centroid_2_count = rows_in_centroid_2.shape[0]

    new_centroids[1]['class'] = centroid_1_class
    new_centroids[2]['class'] = centroid_2_class

    return new_centroids, new_sse, centroid_1_count, centroid_2_count


# main function for kmeans
def kmeans():
    print('************************************K-means****************************************')
    print('')
    training_datasets = ['diabetes_train.csv', 'monks-1.train.txt', 'monks-2.train.txt', 'monks-3.train.txt']
    test_datasets = ['diabetes_test.csv', 'monks-1.test.txt', 'monks-2.test.txt', 'monks-3.test.txt']

    for j in range(0, len(training_datasets)):
        # Training
        training_dataset_file = training_datasets[j]
        print('----------------', training_dataset_file, 'Results---------------')
        # training model
        df = pd.read_csv(training_dataset_file, sep=",")
        train_dataset = pd.DataFrame(df)
        train_dataset.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        columns = list(train_dataset.columns)
        # Randomly selecting centroids
        initial_centroids = pd.DataFrame(train_dataset.sample(2, replace=False))
        initial_centroids.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        initial_centroids.index = range(1, 3)
        centroids = {}
        # Storing centroids in dictionary
        for centroid_index, initial_centroid in initial_centroids.iterrows():
            centroids[centroid_index] = {}
            for column in columns:
                centroids[centroid_index][column] = float(initial_centroid[column])

        print('Initial Centroids')
        for index, row in centroids.items():
            print('centroid:', index)
            print(row)
            print('')
        i = 0
        sse = 0
        threshold = 0
        flag = 0
        # Loop untill flag = 1
        while flag == 0:
            # Calculating new centroids
            new_centroids, new_sse, centroid_1_count, centroid_2_count = create_new_centroids(train_dataset, centroids, sse)
            # Terminating condition using sse
            if i > 0:
                diff = sse - new_sse
                threshold = 0.005*sse
                if diff <= threshold:
                    flag = 1
            centroids = new_centroids
            sse = new_sse
            i += 1
        print('No of iterations completed:', i-1)
        print('Final Centroids')
        for index, row in centroids.items():
            print('centroid:', index)
            print(row)
            if index == 1:
                print('No of rows:', centroid_1_count, ' (', round(100*centroid_1_count/train_dataset.shape[0]), '%)')
            else:
                print('No of rows:', centroid_2_count, ' (', round(100*centroid_2_count/train_dataset.shape[0]), '%)')
            print('')


        # testing
        test_dataset_file = test_datasets[j]
        df = pd.read_csv(test_dataset_file, sep=",")
        test_dataset = pd.DataFrame(df)
        test_dataset.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        columns = list(test_dataset.columns)
        class_column = columns[0]
        test_dataset_count = test_dataset.shape[0]
        predict_correct = 0
        accuracy = 0
        for index, row in test_dataset.iterrows():
            distance_wrt_centroid = {}
            for centroid_index in centroids.keys():
                distance = 0
                for i in range(1, len(columns)):
                    column = columns[i]
                    distance += np.abs(np.square(centroids[centroid_index][column] - row[column]))
                distance = format(np.sqrt(distance), '.2f')
                distance_wrt_centroid[centroid_index] = distance
            centroid_index = min(distance_wrt_centroid, key=distance_wrt_centroid.get)
            predicted_class = centroids[centroid_index][class_column]
            if predicted_class == row[class_column]:
                predict_correct += 1

        print('dt_count:', test_dataset_count)
        print('correct:', predict_correct)
        accuracy = format((predict_correct/test_dataset_count) * 100, '.2f')
        print('accuracy:', accuracy)
        print('')



