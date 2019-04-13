from flask import Flask,render_template,url_for,redirect,flash
from urllib import request
from forms import RegistrationForm, LoginForm, PredictForm
from ploting_continus import *
from corr import *
from model import *
#from important_features import  draw_pic
app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dwde280ba245'
import os
path = "static/images/"


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', posts=posts)

@app.route('/correlation')
def about():
    return render_template('about.html', title='About',posts = features)



@app.route("/predict", methods=['GET','POST'])
def predict():
    form = PredictForm()
    if form.validate_on_submit():
        sex,exang,ca,cp,restecg,slope,thal = form.data['sex'],\
            form.data['exang'],\
            form.data['ca'],\
            form.data['cp'],\
            form.data['restecg'],\
            form.data['slope'],\
            form.data['thal']
        flash("Have Heart Disease" if prediction(sex,exang,ca,cp,restecg,slope,thal)\
            else "No Heart Disease" ,'success')
    return render_template('predict.html', title='predict the heart disease', form=form)

if __name__ == '__main__':

    run_model()

    files = os.listdir(path)
    posts = []
    for file in files:
        tmpDict = {}
        tmpDict['title'] = file.rstrip('.png')
        tmpDict['content'] = 'content'
        tmpDict['date_posted'] = 'April 20, 2018'
        tmpDict['img'] = '/'+path+file
        posts.append(tmpDict)
    #print(posts)
    features=[]
    features.append(get_corr())

    #draw_pic()
    app.run(debug=True)
