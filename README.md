# Group Project: Heart Disease - luckyno.1
In this project, a web app containing three tasks is built. Given the dataset "processed.cleveland.data", the data is cleaned, visualized by groups of age and sex, the potential importance factors are shown, and heart disease is predicted based on the inputs. The attributes are shown as followed:
>1. age
>2. sex (1 = male; 0 = female)
>3. chest pain type (1=typical angin,2=atypical angina,3=non-anginal pain,4=asymptomatic)
>4. resting blood pressure
>5. serum cholestoral in mg/dl
>6. fasting blood sugar > 120 mg/dl
>7. resting electrocardiographic results (0=normal,1=having ST-T wave abnormality (T wave inversions and/or ST elevation or >depression of > 0.05 mV),2=showing probable or definite left
>ventricular hypertrophy by Estes’ criteria)
>8. maximum heart rate achieved
>9. exercise induced angina
>10. oldpeak = ST depression induced by exercise relative to rest
>11. the slope of the peak exercise ST segment
>12. number of major vessels (0-3) colored by flourosopy
>13. thal(Thalassemia): 3 = normal; 6 = fixed defect; 7 = reversable defect
>14. target: have disease or not (0=no,otherwise yes).

## Run web app
In order to run the web app, only ***flaskWebApp.py*** is run as followed:
> git clone https://github.com/GC549739653/9321luckyno.1.git <br>
> cd 9321luckyno.1-master <br>
> python3 flaskWebAPP.py

Paste the IP:port into the browser and the web app will appear immediately.

## Reproduce prediction model
Reomve ***trained_model.sav*** from folder,then run ***model.py***. It will generate a new ***trained_model.sav***.<br>
***trained_model.sav*** can be used like follwoing code:
>model = pickle.load(open("trained_model.sav", 'rb'))
>results = svm.predict([input])
