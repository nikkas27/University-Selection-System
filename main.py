from flask import Flask, render_template, flash, request, url_for, redirect
import sys, csv


# Storing the student's marks from the stored variable to the testing CSV file
def entry(gre,quant,verbal,toefl,gpa):
    j = 0
    # Giving column names
    with open('F:/Cleveland State University/Fall 19/CIS 660/Project/Dataset/Testing_Set/TestFor1690.csv', 'w') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow(['No.','Quant','Verbal','GRE','TOEFL','CGPA'])

    # Storing the value
    with open('F:/Cleveland State University/Fall 19/CIS 660/Project/Dataset/Testing_Set/TestFor1690.csv', 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([j,quant,verbal,gre,toefl,gpa])

    with open('F:/Cleveland State University/Fall 19/CIS 660/Project/Dataset/Testing_Set/TestFor1690.csv', 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([j+1,quant,verbal,gre,toefl,gpa])

    # Reading the test CSV file to know if it inserted correctly..
    with open('F:/Cleveland State University/Fall 19/CIS 660/Project/Dataset/Testing_Set/TestFor1690.csv','r') as newFile:
        newFileReader = csv.reader(newFile)
        for row in newFileReader:
            print(row)


# Function that calls our Machine Learning python files
def USS():
    print("Hello there!!!")

    sys.path.append('F:/Cleveland State University/Fall 19/CIS 660/Project')
    import classify
    print("End of USS")


app = Flask(__name__)


# Loading the main.html page as a home page..
@app.route('/')
def home():
    return render_template('main.html')

# Not implemented
# @app.route('/result/',methods=["GET","POST"])
# def result():
#     return render_template('result.html')

# Loading the probabiliy check page for checking the probability
@app.route('/probability/', methods=["GET","POST"])
def probability():

    error = None
    print(request.method)
    try:
        if request.method == "POST":

            # Fetching the values of the marks entered by the student to the local varibales that is used for storing to the testing CSV file
            attempted_gre = int(request.form['gre'])
            attempted_quant = int(request.form['quant'])
            attempted_verbal = int(request.form['verbal'])
            attempted_toefl = int(request.form['toefl'])
            attempted_gpa = (request.form['gpa'])
            # Checking if the user entered the marks in the correct format
            if attempted_gre <= 340:
                print("Correct....")
                if attempted_verbal <= 170:
                    print("Correct....")
                    if attempted_quant <= 170:
                        print("Correct....")
                        if attempted_toefl <= 120:
                            print("Correct....")

                            entry(attempted_gre,attempted_quant,attempted_verbal,attempted_toefl,attempted_gpa)

                            USS()

            return render_template('probabilitycheck.html', error = error)  # Returning to the same page with some error generated, if any
        return render_template('probabilitycheck.html', error = error)

    except Exception as e:
        error = "Invalid Input. Try Again !"
        print(e)

    return render_template('probabilitycheck.html', error = error)

# Not implemented
# @app.route('/UniversityLists/')
# def unilist():
#     return render_template('UniversityList.html')
#
#
# @app.route('/TopUniversities/')
# def topuni():
#     return render_template('TopUniversities.html')

if __name__ == '__main__':

    app.run(debug=True)

