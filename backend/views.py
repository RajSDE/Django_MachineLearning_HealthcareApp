from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from django.shortcuts import render
import joblib
import numpy as np
from django.http import JsonResponse
from django.http import HttpResponse

# loading trained_model
import joblib as jb
model = jb.load('trained_model')

# ------------------- Rendering Pages -------------------------- #


def home(request):
    return render(request, 'index.html')


def diagnosis(request):
    return render(request, 'diagnosis.html')


def liver(request):
    return render(request, 'liver.html')


def kidney(request):
    return render(request, 'kidney.html')


def heart(request):
    return render(request, 'heart.html')


def diabetes(request):
    return render(request, 'diabetes.html')


def service(request):
    return render(request, 'service.html')

def contact(request):
    return render(request, 'contact.html')

# ------------------- XXXXXXXXXXXXXX -------------------------- #


# ------------------- General Function for All Models -------------------------- #

def ValuePredictor(to_predict_list, size, model_name):
    mdname = str(model_name)
    to_predict = np.array(to_predict_list).reshape(1, size)
    if (size == 7):
        trained_model = joblib.load(rf'{mdname}_model.pkl')
        result = trained_model.predict(to_predict)
    return result[0]

# ------------------- XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX -------------------------- #


# ------------------- Liver Disease -------------------------- #

def lpredictor(request):
    mname = "liver"
    llis = []
    llis = [request.POST.get(i, False) for i in ('Total Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                                                 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio')]

    if (len(llis) == 7):
        result = ValuePredictor(llis, 7, mname)

    if (int(result) == 1):
        return render(request, 'risk.html')
    else:
        return render(request, 'norisk.html')


# ------------------- Disease End --------------------------- #


# ------------------- Kidney Disease ------------------------ #

def kdpredictor(request):
    mname = "kidney"
    klis = []
    klis = [request.POST.get(i, False) for i in (
        'Year', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc')]

    if (len(klis) == 7):
        result = ValuePredictor(klis, 7, mname)

    if (int(result) == 1):
        return render(request, 'risk.html')
    else:
        return render(request, 'norisk.html')

# ------------------- Disease End ------------------------- #


# ------------------- Heart Disease ----------------------- #
warnings.filterwarnings('ignore')


# def HeartPredictor(to_predict_list, size, model_name):
#     heart = pd.read_csv('./MachineLearningModels/DataSets/heartdataNew.csv')
#     labels = heart['target']
#     features = heart.drop(['target'], axis=1)
#     features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=2)
#     logisticRegression = LogisticRegression(solver='lbfgs')
#     logisticRegression.fit(features_train, labels_train)
#     to_predict = np.array(to_predict_list).reshape(1, size)
#     if (size == 7):
#         result = logisticRegression.predict(to_predict)
#     return result[0]


def hdpredictor(request):
    mname = "heart"
    hlis = []
    hlis = [request.POST.get(i, False) for i in (
        'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang')]
    if (len(hlis) == 7):
        result = ValuePredictor(hlis, 7, mname)
        # result = HeartPredictor(hlis, 7, mname)

    if (int(result) == 1):
        return render(request, 'risk.html')
    else:
        return render(request, 'norisk.html')

# ------------------- Heart End -------------------------- #


# ------------------- Diabetes Disease ----------------------- #

def dbpredictor(request):
    dblis = []
    dblis.append(request.POST['Pregnancies'])
    dblis.append(request.POST['Present_Price'])
    dblis.append(request.POST['BloodPressure'])
    dblis.append(request.POST['BMI'])
    dblis.append(request.POST['DiabetesPedigreeFunction'])
    dblis.append(request.POST['Age'])
    if (len(dblis) == 6):
        result = DiabetesValuePredictor(dblis, 6)

    if (int(result) == 1):
        return render(request, 'risk.html')
    else:
        return render(request, 'norisk.html')


# Sample Machine Learning Code for diabetes prediction

diabetes_dataset = pd.read_csv('./diabetes.csv')


def DiabetesValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    # to_predict = np.array(to_predict_list).reshape(1,size)

    if (size == 6):
        # trained_model = joblib.load(r'diabetes_model.pkl')
        # trained_model = pd.(r'diabetes_model.pkl')
        # result = trained_model.predict(to_predict)
        std_data = scaler.transform(to_predict)
        result = classifier.predict(std_data)
    return result[0]

# ------------------- Diabetes End ----------------------- #


# self assessment
def assessment(request):

    diseaselist = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
                   'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
                   'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
                   'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
                   'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
                   'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']

    # symptomslist = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
    #                 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination',
    #                 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    #                 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
    #                 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    #                 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    #                 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    #                 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    #                 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    #                 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    #                 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    #                 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    #                 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    #                 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    #                 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    #                 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    #                 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    #                 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    #                 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    #                 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    #                 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    #                 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
    #                 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    #                 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    #                 'yellow_crust_ooze']

    symptomslist = ['Itching', 'Skin Rash', 'Nodal Skin Eruptions', 'Continuous Sneezing', 'Shivering', 'Chills', 'Joint Pain', 'Stomach Pain', 'Acidity', 'Ulcers On Tongue', 'Muscle Wasting', 'Vomiting', 'Burning Micturition', 'Spotting  urination', 'Fatigue', 'Weight Gain', 'Anxiety', 'Cold Hands And Feets', 'Mood Swings', 'Weight Loss', 'Restlessness', 'Lethargy', 'Patches In Throat', 'Irregular Sugar Level', 'Cough', 'High Fever', 'Sunken Eyes', 'Breathlessness', 'Sweating', 'Dehydration', 'Indigestion', 'Headache', 'Yellowish Skin', 'Dark Urine', 'Nausea', 'Loss Of Appetite', 'Pain Behind The Eyes', 'Back Pain', 'Constipation', 'Abdominal Pain', 'Diarrhoea', 'Mild Fever', 'Yellow Urine', 'Yellowing Of Eyes', 'Acute Liver Failure', 'Fluid Overload', 'Swelling Of Stomach', 'Swelled Lymph Nodes', 'Malaise', 'Blurred And Distorted Vision', 'Phlegm', 'Throat Irritation', 'Redness Of Eyes', 'Sinus Pressure', 'Runny Nose', 'Congestion', 'Chest Pain', 'Weakness In Limbs', 'Fast Heart Rate', 'Pain During Bowel Movements', 'Pain In Anal Region', 'Bloody Stool', 'Irritation In Anus', 'Neck Pain', 'Dizziness', 'Cramps', 'Bruising', 'Obesity', 'Swollen Legs', 'Swollen Blood Vessels', 'Puffy Face And Eyes', 'Enlarged Thyroid',
                    'Brittle Nails', 'Swollen Extremeties', 'Excessive Hunger', 'Extra Marital Contacts', 'Drying And Tingling Lips', 'Slurred Speech', 'Knee Pain', 'Hip Joint Pain', 'Muscle Weakness', 'Stiff Neck', 'Swelling Joints', 'Movement Stiffness', 'Spinning Movements', 'Loss Of Balance', 'Unsteadiness', 'Weakness Of One Body Side', 'Loss Of Smell', 'Bladder Discomfort', 'Foul Smell Of urine', 'Continuous Feel Of Urine', 'Passage Of Gases', 'Internal Itching', 'Toxic Look (typhos)', 'Depression', 'Irritability', 'Muscle Pain', 'Altered Sensorium', 'Red Spots Over Body', 'Belly Pain', 'Abnormal Menstruation', 'Dischromic  Patches', 'Watering From Eyes', 'Increased Appetite', 'Polyuria', 'Family History', 'Mucoid Sputum', 'Rusty Sputum', 'Lack Of Concentration', 'Visual Disturbances', 'Receiving Blood Transfusion', 'Receiving Unsterile Injections', 'Coma', 'Stomach Bleeding', 'Distention Of Abdomen', 'History Of Alcohol Consumption', 'Fluid Overload', 'Blood In Sputum', 'Prominent Veins On Calf', 'Palpitations', 'Painful Walking', 'Pus Filled Pimples', 'Blackheads', 'Scurring', 'Skin Peeling', 'Silver Like Dusting', 'Small Dents In Nails', 'Inflammatory Nails', 'Blister', 'Red Sore Around Nose', 'Yellow Crust Ooze']
    print(len(symptomslist))

    sortedsymptoms = sorted(symptomslist)
    user = {
        "name": "Raj Kumar",
        "age": 23
    }
    if request.method == 'GET':

        return render(request, 'assessmentPage.html', {"list2": sortedsymptoms, "user": user})

    elif request.method == 'POST':

        # access you data by playing around with the request.POST object

        inputno = int(request.POST["noofsym"])
        print(inputno)
        if (inputno == 0):
            return JsonResponse({'predicteddisease': "none", 'confidencescore': 0})

        else:

            psymptoms = []
            psymptoms = request.POST.getlist("symptoms[]")

            print(psymptoms)

            """      #main code start from here...
        """

            testingsymptoms = []
            # append zero in all coloumn fields...
            for x in range(0, len(symptomslist)):
                testingsymptoms.append(0)

            # update 1 where symptoms gets matched...
            for k in range(0, len(symptomslist)):

                for z in psymptoms:
                    if (z == symptomslist[k]):
                        testingsymptoms[k] = 1

            inputtest = [testingsymptoms]

            print(inputtest)

            predicted = model.predict(inputtest)
            print("Predicted disease is :", predicted[0])
            y_pred_2 = model.predict_proba(inputtest)
            confidencescore = y_pred_2.max() * 100
            print("Confidence score of \"{0}\" is {1} %".format(
                predicted[0], confidencescore))

            confidencescore = format(confidencescore, '.0f')
            predicted_disease = predicted[0]

            Rheumatologist = ['Osteoarthristis', 'Arthritis']

            Cardiologist = ['Heart attack',
                            'Bronchial Asthma', 'Hypertension ']

            ENT_specialist = [
                '(vertigo) Paroymsal  Positional Vertigo', 'Hypothyroidism']

            Orthopedist = []

            Neurologist = [
                'Varicose veins', 'Paralysis (brain hemorrhage)', 'Migraine', 'Cervical spondylosis']

            Allergist_Immunologist = ['Allergy', 'Pneumonia', 'AIDS',
                                      'Common Cold', 'Tuberculosis', 'Malaria', 'Dengue', 'Typhoid']

            Urologist = ['Urinary tract infection',
                         'Dimorphic hemmorhoids(piles)']

            Dermatologist = ['Acne', 'Chicken pox',
                             'Fungal infection', 'Psoriasis', 'Impetigo']

            Gastroenterologist = ['Peptic ulcer diseae', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Gastroenteritis', 'Hepatitis E',
                                  'Alcoholic hepatitis', 'Jaundice', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Diabetes ', 'Hypoglycemia']

            if predicted_disease in Rheumatologist:
                consultdoctor = "Rheumatologist"

            if predicted_disease in Cardiologist:
                consultdoctor = "Cardiologist"

            elif predicted_disease in ENT_specialist:
                consultdoctor = "ENT specialist"

            elif predicted_disease in Orthopedist:
                consultdoctor = "Orthopedist"

            elif predicted_disease in Neurologist:
                consultdoctor = "Neurologist"

            elif predicted_disease in Allergist_Immunologist:
                consultdoctor = "Allergist/Immunologist"

            elif predicted_disease in Urologist:
                consultdoctor = "Urologist"

            elif predicted_disease in Dermatologist:
                consultdoctor = "Dermatologist"

            elif predicted_disease in Gastroenterologist:
                consultdoctor = "Gastroenterologist"

            else:
                consultdoctor = "other"

            request.session['doctortype'] = consultdoctor

            # patientusername = request.session['patientusername']
            # puser = User.objects.get(username=patientusername)

            # saving to database.....................

            # patient = puser.patient
            # patient = 'Raj Kumar'
            # diseasename = predicted_disease
            # no_of_symp = inputno
            # symptomsname = psymptoms
            # confidence = confidencescore

            # diseaseinfo_new = diseaseinfo(patient=patient,diseasename=diseasename,no_of_symp=no_of_symp,symptomsname=symptomsname,confidence=confidence,consultdoctor=consultdoctor)
            # diseaseinfo_new.save()

            # request.session['diseaseinfo_id'] = diseaseinfo_new.id

            # print("disease record saved sucessfully.............................")

            return JsonResponse({'predicteddisease': predicted_disease, 'confidencescore': confidencescore, "consultdoctor": consultdoctor})
