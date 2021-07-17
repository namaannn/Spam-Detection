from flask import Flask,render_template,send_from_directory,request
import pickle

model=pickle.load(open('spam.pkl','rb'))
tfd=pickle.load(open('tfid.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home3.html')
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        msg=[message]
        ms=tfd.transform(msg).toarray()
        output=model.predict(ms)
        return render_template('result2.html',prediction=output)

if __name__=='__main__':
    app.run(debug=True)
        