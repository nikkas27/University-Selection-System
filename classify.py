import pickle
from acc_check import *
from collections import Counter
import csv

prob_accept_rf, prob_reject_rf = {}, {}


# Function that is used to get the value of the probabability percent of getting an admit
def Probability(predicted, probability, accept, reject):
    if predicted == 1:
        for k in range(len(probability)):
            # print("Before checking",probability[k][0])
            if probability[k][0] > probability[k][1]:
                # print(".......")
                # print("Checking the if statement",probability[k][0])
                accept[k] = probability[k][0]
                return accept[k]*100
            else:
                # print(".......******")
                # print(probability[k][0])
                accept[k]=probability[k][1]
                return accept[k]*100
    else:
        for k in range(len(probability)):
            if probability[k][0] > probability[k][1]:
                print(probability[k][0])
                reject[k]=probability[k][0]
                return reject[k]*100
            else:
                print(probability[k][1])
                reject[k]=probability[k][1]
                return reject[k]*100


# Reading the test data file which contains the students marks
test_data = pd.read_csv('F:/Cleveland State University/Fall 19/CIS 660/Project/Dataset/Testing_Set/TestFor1690.csv')
test_data.dropna(inplace=True)  # Dropping any values that is NULL

# Storing the feature values to the testing variables
test_gre = test_data.values[:, 3]
test_quant = test_data.values[:, 1]
test_verbal = test_data.values[:, 2]
test_toefl = test_data.values[:, 4]
test_cgpa = test_data.values[:, 5]

# Converting the testing values to the normalized form
test_gre = (test_gre - min_gre)/(max_gre-min_gre)
test_quant = (test_quant - min_quant)/(max_quant-min_quant)
test_verbal = (test_verbal - min_verbal)/(max_verbal-min_verbal)
test_toefl = (test_toefl - min_toefl)/(max_toefl-min_toefl)

# Converting the gpa value in the range of 0-1
test_score = cgpa_convert(test_cgpa)

# Converting the regular data to machine understandable form with specific weight.
test_data['Quant'] = test_quant
test_data['Verbal'] = test_verbal
test_data['GRE'] = test_gre
test_data['TOEFL'] = test_toefl
test_data['CGPA'] = test_cgpa

Test_independent_var = test_data.values[-1:, 1:6]  # Selecting the testing features that will be used for prediction

for u in uni:  # Iterating through each university model and predicting whether that particular student will get admit to that particular university
    print("------------------{0}------------------".format(u))
    for l in range(1):

        # Loading the Random Forest model saved while training
        filename = 'F:/Cleveland State University/Fall 19/CIS 660/Project/Models/rf_acc_avg_{0}.sav'.format(u)
        loaded_model = pickle.load(open(filename, 'rb'))
        print("Model loaded")
        # -------------------------------------------Testing phase starts here..------------------------------------

        # Predicting based on the given actual student's data
        predicted_test = loaded_model.predict(Test_independent_var)

        # Predicting the probability for that particular university
        predicted_test_prob = loaded_model.predict_proba(Test_independent_var)

        # Calling the function to get the value of the admit probability
        prob_result = Probability(predicted_test,predicted_test_prob,prob_accept_rf,prob_reject_rf)

        # If the student got an admit then save that probability percent to the dict
        if predicted_test == 1:
            prob_accept_rf[u] = prob_result

# Rearranging Uni and probability with the most at the first in order
prob_accept_rf = dict(Counter(prob_accept_rf).most_common())
print(prob_accept_rf) # printing the whole dict after sorting

# Saving this prediction to a csv file for future lookup..
for i in prob_accept_rf:
    with open('F:/Cleveland State University/Fall 19/CIS 660/Project/Dataset/UniProb.csv', 'a') as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow([i, prob_accept_rf[i]])
            # print("DOne!")
