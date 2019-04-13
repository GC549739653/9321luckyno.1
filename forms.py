from flask_wtf import FlaskForm
from wtforms import *
from wtforms.validators import *


class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class PredictForm(FlaskForm):
    age = IntegerField("Age", validators=[NumberRange(min=1,max=100)])
    sex = SelectField("Sex", choices=[(1,"Male"),(0,"Female")],coerce=int)
    cp = SelectField("Chest Pain Type", choices=[(1,"Typical angin"),
        (2,"atypical angina"),
        (3,"non-anginal pain"),
        (4,"asymptomatic")],coerce=int)
    trestbps = IntegerField("Resting Blood Pressure", validators=[DataRequired(),NumberRange(min=80,max=200)])
    chol = IntegerField("Serum Cholestoral in mg/dl", validators=[DataRequired(),NumberRange(min=100,max=600)])
    fbs = SelectField("Fasting Blood Sugar > 120 mg/dl?",choices=[(1,"Yes"),
        (0,"No")],coerce=int)
    restecg = SelectField("Resting Electrocardiographic Result",choices=[(0,"Normal"),
        (1,"Having ST-T Wave Abnormality"),
        (2,"Showing Probable or Definite Left Ventricular Hypertrophy")],coerce=int)
    thalach = IntegerField("Maximum Heart Rate Achieved", validators=[DataRequired(),NumberRange(min=50,max=250)])
    exang = SelectField("Exercise Induced Angina", choices=[(1,"Yes"),
        (0,"No")],coerce=int)
    oldpeak = DecimalField("ST Depression Indec. Ex.", validators=[DataRequired()])
    slope = IntegerField("Slope of Peak Exercise ST", validators=[DataRequired()])
    ca = SelectField("Number of Major Vessel", choices=[(0,"0"),
        (1,"1"),
        (2,"2"),
        (3,"3")],coerce=int)
    thal = SelectField("Thalassemia?", choices=[(3,"Normal"),
        (6,"Fixed defect"),
        (7,"Reversable defect")],coerce=int)
    submit = SubmitField("Submit")