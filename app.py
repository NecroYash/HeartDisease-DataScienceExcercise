
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('saved_KNNmodel.sav')

def predict(input_features):

    prediction = model.predict(input_features)
    return prediction

#judul
st.title('UCI Heart Disease Prediction')

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:

  #data fitur yang perlu di isi
  age = st.slider('Age', 0.0, 100.0, 50.0, step = 1.0)
  sex = st.selectbox('Sex', ['Male', 'Female'])
  cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina','Non-Anginal Pain','Asymptomatic'])
  trestbps = st.slider('Resting Blood Pressure', 50.0, 180.0, 80.0, step = 1.0)
  chol = st.slider('serum cholestoral', 100.0, 250.0, 150.0, step = 1.0)
  fbs = st.selectbox('Fasting Blood Sugar', ['Yes', 'No'])
  restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', ' Left Ventricular Hypertrophy'])
  thalach = st.slider('Maximum Heart Rate Achieved', 80.0, 220.0, 100.0, step = 1.0)
  exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
  oldpeak = st.slider('Maximum Heart Rate Achieved', -3.0, 7.0,0.0, step = 0.1)

  if st.button('Predict'):
      # Konversi input ke dalam format yang dapat digunakan oleh model
      input_features = pd.DataFrame({
          'age': [age],
          'sex': [1.0 if sex == 'Male' else 0.0],  # Misalnya, 1 untuk Male dan 0 untuk Female
          'cp' : [1.0 if cp == 'Typical Angina' else 2.0 if cp == 'Atypical Angina' else 3.0 if cp == 'Non-Anginal Pain' else 4.0],
          'trestbps' : [trestbps],
          'chol' : [chol],
          'fbs' : [1.0 if fbs == 'Yes' else 0.0],
          'restecg' : [0.0 if restecg == 'Normal' else 1.0 if restecg == 'ST-T Wave Abnormality' else 2.0],
          'thalach' : [thalach],
          'exang' : [1.0 if exang == 'Yes' else 0.0],
          'oldpeak' : [oldpeak]
      })

      # Lakukan prediksi
      result = predict(input_features)

      # Konversi hasil prediksi ke dalam label yang lebih bermakna
      if result >= 1.0:
          prediction_label = 'Penyakit Jantung Terdeteksi'
      else:
          prediction_label = 'Tidak Terdeteksi Penyakit Jantung'

      st.success(f'The prediction is: {prediction_label}')

with tab2:

  st.header("Predict multiple data:")

  st.write("")
  st.download_button("Download CSV Example", data='sample_heart_disease_parameteers.csv', file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    for prediction in prediction_arr:
      if prediction >= 1:
        result = "Penyakit Jantung Terdeteksi"
      else:
        result = "Tidak Terdeteksi Penyakit Jantung"
      result_arr.append(result)

    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)
