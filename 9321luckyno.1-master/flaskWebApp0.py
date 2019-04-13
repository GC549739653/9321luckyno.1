# from flask import Flask,render_template,url_for,redirect,flash
# from urllib import request
# from forms import RegistrationForm, LoginForm
# from ploting_continus import *
# app = Flask(__name__)
# app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dwde280ba245'
# import os
# path = "static/images/"
#
# # posts = [
# #     {
# #         'title': 'Graph 1',
# #         'content': 'First content',
# #         'date_posted': 'April 20, 2018',
# #         'img': '/static/resting electrocardiographic results.png'
# #     },
# #     {
# #         'title': 'Graph 2',
# #         'content': 'Second content',
# #         'date_posted': 'April 21, 2018',
# #         'img': '/continual/pie%chart%-%oldpeak%=%ST%depression%induced%by%exercise%relative%to%rest.png'
# #     }
# # ]
#
# @app.route('/')
# @app.route('/home')
# def home():
#     return render_template('home.html', posts=posts)
#
# @app.route('/about')
# def about():
#     return render_template('about.html', title='About')
#
# @app.route("/register", methods=['GET', 'POST'])
# def register():
#     form = RegistrationForm()
#     if form.validate_on_submit():
#         flash(f'Account created for {form.username.data}!', 'success')
#         return redirect(url_for('home'))
#     return render_template('predict.html', title='Register', form=form)
#
# @app.route("/predict", methods=['GET','POST'])
# def predict():
#     form = PredictForm()
#     if form.validate_on_submit():
#         flash(form.data['age'],'success')
#     return render_template('predict.html', title='Register', form=form)
#
#
#
# if __name__ == '__main__':
#
#     files = os.listdir(path)
#     posts = []
#     for file in files:
#         tmpDict = {}
#         tmpDict['title'] = file.rstrip('.png')
#         tmpDict['content'] = 'content'
#         tmpDict['date_posted'] = 'April 20, 2018'
#         tmpDict['img'] = '/'+path+file
#         posts.append(tmpDict)
#     print(posts)
#     app.run(debug=True)
