from flask import Flask,render_template,request
import pandas as pd
import pickle

application=Flask(__name__)
app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_point():
    if request.method=='GET':
        return render_template('index.html')
    else:
        sepal_length=float(request.form.get('sepal_length'))
        sepal_width=float(request.form.get('sepal_width'))
        petal_length=float(request.form.get('petal_length'))
        petal_width=float(request.form.get('petal_width'))
        x_new=pd.DataFrame([sepal_length,sepal_width,petal_length,petal_width]).T
        x_new.columns=['sepal_length','sepal_width','petal_length','petal_width']
        with open('notebook/pipe.pkl','rb') as file1:
            pipe=pickle.load(file1)
        x_pre=pipe.transform(x_new)
        x_pre=pd.DataFrame(x_pre,columns=x_new.columns)

        with open('notebook/model.pkl','rb') as file2:
            model=pickle.load(file2)

        with open('notebook/Labelenc.pkl','rb') as file3:
            le=pickle.load(file3)

        pred=model.predict(x_pre)

        pred_lb=le.inverse_transform(pred)[0]
        prob=model.predict_proba(x_pre).max()

        prediction=f'{pred_lb} with probability: {prob:.4f}'

        return render_template('index.html',prediction=prediction)




if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)



