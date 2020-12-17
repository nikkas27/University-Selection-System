import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.metrics import accuracy_score
from sklearn import metrics
import plotly.graph_objects as go
import chart_studio.plotly as py
import pickle
import plotly.express as px


def cgpa_convert(score):  # Function created to convert the CGPA into the range of 0-1
    for cc in range(len(score)):
        # print(cgpa_score[c])
        if score[cc] < 10:
            score[cc] = score[cc] / 10
        elif score[cc] > 10:
            score[cc] = score[cc] / 100
    # print(score)
    return score


# Parsing through each universities excel file generated from the full university dataset..
uni = ['Cleveland_State_University', 'Stanford_University', 'UCBerkley_University', 'Arizona_State_University',
       'CMU_University', 'CSU_LongBeach', 'CSU_LosAngeles_University', 'SanJoseStateUniversity', 'njit',
       'uni_of_illinois_chicago_uni','CU','MTU']

for u in uni:
    # print("-------------------------{0}---------------------------------------".format(u))
    # Reading the Excel file for a particular university
    i = pd.read_excel("F:/Cleveland State University/Fall 19/CIS 660/Project/Dataset/Training_Set/{0}.xlsx".format(u))
    # Dropping the unnecessary columns that are not needed for the training of the models..
    i.drop(["University Name"], 1, inplace=True)
    i.drop(["Branch"], 1, inplace=True)
    i.drop(["Year"], 1, inplace=True)
    i.drop(["AWA"], 1, inplace=True)
    i.drop(["UG college"], 1, inplace=True)
    i.drop(["UG major"], 1, inplace=True)
    i.drop(["Experience"], 1, inplace=True)
    i.drop(["Papers"], 1, inplace=True)

    if u == 'CMU_University':  # Removing two columns that were generated while dividing the full dataset file to each particular university
        i.drop(["Unnamed: 14"], 1, inplace=True)
        i.drop(["Unnamed: 15"], 1, inplace=True)

    # Extra feature to add to increase the prediction accuracy....

    cleanup_nums = {"Result": {"Admit": 1, "Reject": 0}}  # Converting the Admits and Rejects to 0 and 1 that will be useful for comparing the outputs
    i.replace(cleanup_nums, inplace=True)

    i.dropna(inplace=True)  # Dropping any values containing a null value from the dataset..
    # print(i.isnull().any())

    # storing each values into the variables
    quant_score = i.values[:, 1]
    verbal_score = i.values[:, 2]
    gre_score = i.values[:, 3]
    toefl_score = i.values[:, 4]
    cgpa_score = i.values[:, 5]

    # calling the cgpa convert function to convert the given cgpa in the range of 0-1
    train_cgpa = cgpa_convert(cgpa_score)

    # giving the min and max limit of gre
    min_gre = 260
    max_gre = 340

    # giving the min and max limit of quant
    min_quant = 130
    max_quant = 170

    # giving the min and max limit of verbal
    min_verbal = 130
    max_verbal = 170

    # giving the min and max limit of toefl
    min_toefl = 0
    max_toefl = 120

    # normalizing to convert the values in the range of 0-1
    for g in range(len(gre_score)):
        gre_score[g] = (gre_score[g] - min_gre) / (max_gre - min_gre)

    for q in range(len(quant_score)):
        quant_score[q] = (quant_score[q] - min_quant) / (max_quant - min_quant)

    for v in range(len(verbal_score)):
        verbal_score[v] = (verbal_score[v] - min_quant) / (max_quant - min_quant)

    for t in range(len(toefl_score)):
        toefl_score[t] = (toefl_score[t] - min_toefl) / (max_toefl - min_toefl)

    # Storing these converted values to the dataset
    i['Quant'] = quant_score
    i['Verbal'] = verbal_score
    i['GRE'] = gre_score
    i['TOEFL'] = toefl_score
    i['CGPA'] = train_cgpa

    # ------------------------------Training phase starts here....----------------------------------

    Train_independent_var = i.values[:, 1:6]  # contains all the features that are going to be the input for our models
    Train_target_var = i.values[:, 0]  # contains the target values i.e. 0/1 whether got admit or not

    # initializing the list values with 0
    rf_acc_avg, nb_acc_avg, knn_acc_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    algo_list = [rf_acc_avg, nb_acc_avg, knn_acc_avg]
    algo_name_list = ['rf_acc_avg', 'nb_acc_avg', 'knn_acc_avg']

    # input parameters for the models
    cls_wghts = 'balanced'  #Weights associated with teh class
    mx_dpth = 5     # Depth of the tree
    crit = 'entropy'    # Function to use to measure the quality of split
    estimators = 75  # No. of trees in the forest
    rd_state = 200
    algo = 'auto'    # Algorihtm used to calculate the neighbors - ball_tree, kd_tree, brute-force, auto
    neighbor = 7    # No. of neighbors

    max_rf, max_nb, max_knn = 0, 0, 0

    for a in range(10):  # iterating to get the best model prediction out of 20 testings..
        # print("----------------------------------{0}------------------------------".format(a))
        # ------------------------------------Random Forest------------------------------
        # Splitting the data with 80% for training and 20 for testing purpose
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Train_independent_var, Train_target_var,
                                                                             test_size=0.2)
        # Building the model with the given parameters above
        rf_model = RandomForestClassifier(max_depth=mx_dpth, n_estimators=estimators, random_state=rd_state,
                                          criterion=crit)
        rf_model.fit(X_train, Y_train)

        # Predicting based on the given testing sets
        predicted_rf = rf_model.predict(X_test)

        # Calculating the accuracy score by comparing the output generated by the model vs the actual results
        Accuracy_Score_rf = accuracy_score(Y_test, predicted_rf)

        rf_acc_avg[a] = Accuracy_Score_rf  # Storing these accuracy for comparing in future for the best accuracy output..

        if rf_acc_avg[a] > max_rf:
            max_rf = rf_acc_avg[a]  # Storing this accuracy to a temp variable

            # Saving this particular model which will be used in future for our actual testing set
            filename = 'F:/Cleveland State University/Fall 19/CIS 660/Project/Models/{0}_{1}.sav'.format(
                algo_name_list[0], u)
            pickle.dump(rf_model, open(filename, 'wb'))

            print("Pickle saved...")

        # print(i.columns)
        # print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, predicted_rf))  #Calculating the mean absolute error to see how much wrong our model predicted
        # print('Mean Squared Error:', metrics.mean_squared_error(Y_test, predicted_rf))
        # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, predicted_rf)))

        # Showing the importance given to each of the feature by the Random Forest for that particular training set..
        importance_rf = rf_model.feature_importances_
        c = importance_rf * 100
        importance_rf = pd.DataFrame(c, index=i.columns[1:6],
                                     columns=["Importance"])

        # print(importance_rf)
        # ------------------------------------K Nearest Neighbor------------------------------
        # print("--------------K nearrest neighbor---------------")
        # Building the model with the given parameters above
        clf_knn = neighbors.KNeighborsClassifier(algorithm=algo, n_neighbors=neighbor)

        clf_knn.fit(X_train, Y_train)

        # Predicting based on the given testing sets
        predicted_knn = clf_knn.predict(X_test)

        # Calculating the accuracy score by comparing the output generated by the model vs the actual results
        Accuracy_Score_knn = accuracy_score(Y_test, predicted_knn)

        knn_acc_avg[a] = Accuracy_Score_knn  # Storing these accuracy for comparing in future for the best accuracy output..

        if knn_acc_avg[a] > max_knn:
            max_knn = knn_acc_avg[a]  # Storing this accuracy to a temp variable

            # Saving this particular model which will be used in future for our actual testing set
            filename = 'F:/Cleveland State University/Fall 19/CIS 660/Project/Models/{0}_{1}.sav'.format(
                algo_name_list[1], u)
            pickle.dump(clf_knn, open(filename, 'wb'))

        print("Pickle saved...")

        # ------------------------------------Naive Bayes------------------------------
        # print("--------------- NB------------------")

        # Splitting the data with 80% for training and 20 for testing purpose
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(Train_independent_var, Train_target_var,
                                                                             test_size=0.2)

        # Building the model
        gnb_model = GaussianNB()

        gnb_model.fit(X_train, y_train)

        # Predicting based on the given testing sets
        predicted_gnb = gnb_model.predict(X_test)

        # Calculating the accuracy score by comparing the output generated by the model vs the actual results
        Accuracy_Score_gnb = accuracy_score(y_test, predicted_gnb)

        nb_acc_avg[a] = Accuracy_Score_gnb  # Storing these accuracy for comparing in future for the best accuracy output..

        if nb_acc_avg[a] > max_nb:
            max_nb = nb_acc_avg[a]  # Storing this accuracy to a temp variable

            # Saving this particular model which will be used in future for our actual testing set
            filename = 'F:/Cleveland State University/Fall 19/CIS 660/Project/Models/{0}_{1}.sav'.format(
                algo_name_list[2], u)
            pickle.dump(gnb_model, open(filename, 'wb'))

            print("Pickle saved...")

    # showing the model accuracy vary with different parameters..
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y=rf_acc_avg,
                             mode='lines+markers',
                             name='Random Forest Accuracy'))
    fig.add_trace(go.Scatter(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y=nb_acc_avg,
                             mode='lines+markers',
                             name='Naive Bayes Accuracy'))
    fig.add_trace(go.Scatter(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y=knn_acc_avg,
                             mode='lines+markers',
                             name='K Nearest Neighbor Accuracy'))
    # fig.show()
    # py.image.save_as(fig, filename='Acc_{0}_maxdepth={1}_estimators={2}_randomstate={3}_criterion={4}_algo{5}_neighbors{6}.png'.format(u,mx_dpth,estimators,rd_state,crit,algo,neighbor))

    # printing the maximum accuracy of all 20 iterations..
    # print("*********************Maximum accuracy of RF :", np.max(rf_acc_avg))
    # print("*********************Maximum accuracy of NB :", np.max(nb_acc_avg))
    # print("*********************Maximum accuracy of KNN :", np.max(knn_acc_avg))
