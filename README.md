# An-automatic-Micro-expression-System-of-Detection-and-Recognition

--- Code information ---
face_reg.py: recognize the coordinate of the face
cutFace.py: remove the background of the pictures to improve the accuracy
rotate.py: if the amount of data is not enough, use this one to enhance the amount of data
detect_micro_model.py: to distinguish micro and non-micro pictures in the database
detect_expression_model.py: we seperate the database into three categories which is negative, positive, surprise
testing_micro.py: test the model produced by detect_micro_model.py
testing_expression.py: test the model produced by detect_expression_model.py

We use machine learning to train the CASMEII database, producing the model with accuracy of 77%.
This number is higher than human identification by naked eyes which is approximately 45%.
