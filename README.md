# University-Selection-System
Studying abroad comes with a lot of challenges and risks which involve huge investments of finance and time. This uncertainty needs to be addressed with a statistical model which can help students to narrow down on a set of Universities from a broader spectrum. This model accepts student information and based on data, which will be compared against university cut-off admission criteria of previous term to determine probability of a student to get an admit from the university to which student will be applying.

# Main.py 
It consist of the interface between the Flask and backend Python code. It defines the main.html, probabilitycheck.html, UniversityList.html, TopUniversities.html. 

# acc_check.py
It consist of the implementation of the Machine Learning algorithms by preprocessing the previous year students data of GRE, Quant, Verbal, TOEFL/ILETS, GPA, Percentage. It generates the graph indicating the best algorithm that can be used for the testing phase. The algorithms used for the training phase are Random Forest, K Nearest Neighbors, Naive Bayes. For the graph plotting, I have used Plotly library that displays the graph online and also is being saved locally.


