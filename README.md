> **For a detailed explanation of the system architecture, models, and codebase, please see our [Full Project Documentation (DOCS.md)](DOCS.md).**

![Kidney](https://media.tenor.com/gw6jgBdh0JwAAAAC/rambo-worst-ever.gif)

![System Data Flow](screenshot/system%20data%20flow.png)

# Project

## To get started

1. Clone the repo
   ```
   git clone https://github.com/sampathvenur/Project.git
   cd Project
   ```
[python 3.12.10](https://www.python.org/downloads/release/python-31210/) 👈 Download and install python <br>
2. Install Python dependencies
   ```
   pip install -r requirements.txt
   ```


(Open two terminal sessions to run step 3 and step 4) <br>
3. Run the Python API
   ```
   python -m uvicorn api:app --reload
   ```
 [go](https://go.dev/doc/install) 👈 Download and install go <br>
4. Run the Go backend
   ```
   go run main.go
   ```


5. Open your browser and go to ```http://localhost:8080``` to access the web interface

## Models till 04-10-2025

1. CNN model
   ```
   60/60 [==============================] - 5s 89ms/step - loss: 0.3823 - accuracy: 0.8500
    [0.38234245777130127, 0.8500000238418579]
   ```

2. SVM model
   ```
   Accuracy :  0.7966666666666666
   ```

3. Classifier model
   ```
   stone_type_classifier.pkl
   ```

4. Yolo image classifier model
   ```
   check runs/ dir for result sheets
   ```

5. Resnet classifier model
   ```
   stone_type_classifier_resnet.pkl
   ```

   # _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://media2.giphy.com/media/BPJmthQ3YRwD6QqcVD/giphy.gif)
