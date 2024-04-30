import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import random
import time
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, r2_score
import pickle
import base64

app_mode = st.sidebar.selectbox('Select page:',['01 Introduction','02 Visualization', '03 Prediction','04 Explainable AI','05 MLFlow Tracking'])
if app_mode == '01 Introduction':
  image_movie = Image.open('Title.jpeg')
  st.image(image_movie, width=400)

  st.title("What are the Key Influencers of :orange[_Obesity_] ? ")
  st.subheader("A predictive analysis for health recommendation purposes", divider='rainbow')
  # app_mode = st.sidebar.selectbox('Select Page',['Introduction'])

  st.markdown('##### WHY THIS TOPIC❓')
  st.markdown('Obesity, which causes physical and mental problems, is a global health problem with serious consequences. ')
  st.markdown('The prevalence of obesity is increasing steadily, and therefore, this project is needed to examine the influencing factors of obesity and to predict the occurrence of the condition according to these factors.')

  st.markdown("##### OUR GOAL 🎯 ")
  st.markdown("With our project, we seek to identify the highest influential factor on individual’s obesity levels for health recommendation purposes. ")
  st.markdown("The main objectives of evaluation is focusing on the personal habits, family history, eating habits and physical activity frequency. ")
  st.markdown("Since our research are conducting with health recommendation purposes, we will avoid analyzing on the individuals' traits such as age, height, gender but rather focuses on personal behaviors that could be changed. ")

  st.markdown("#### BIASES 🧐")
  st.markdown("BIASE #1: Since we are not focusing on the individuals' traits such as age, height and gender, it might have a impact on our final prediction results. Because: they are still key factors relevant to obesity levels.")
  st.markdown("BIASE #2: This study only collects data from three countries which limits our ability to apply the final prediction results on a global perspective.")

  st.markdown('##### OUR DATA 📊')
  st.markdown("This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia")
  st.markdown('Our data contains 17 attributes and 2111 records')
  st.markdown("As described by the dataset provider, 77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform.")


  st.markdown("##### Explaination of KEY VARIABLES abbreviations 📓")
  st.markdown("- PERSONAL HABITS ")
  st.markdown("CH2O: How much water do you drink daily?")
  st.markdown("SMOKE: Do you smoke? ")
  st.markdown("TECH: How much time do you use technological devices?")
  st.markdown("CALC: How often do you drink alcohol?")
  st.markdown("MTRANS: Which transportation do you usually use?")
  st.markdown("SCC: Do you monitor the calories you eat daily")

  st.markdown("- EATING HABITS")
  st.markdown("FAVC : Do you eat high caloric food frequently?")
  st.markdown("FCVC : Do you usually eat vegetables in your meals?")
  st.markdown("NCP: 'How many main meals do you have daily?")
  st.markdown("CAEC: DO you eat any food between meals?")

  st.markdown('- PHYSICAL ACTIVITY')
  st.markdown("FAF: How often do you have physical activity?")

  st.markdown('- FAMILY HISTORY')
  st.markdown("FHWO: Family history with overweight")
  st.markdown("OL: Obesity Level")


  st.markdown("### Description of Data")
  df = pd.read_csv("FINAL PROJECT.csv")
  st.dataframe(df.describe())
  st.markdown("🔍 Observation: Based on the description of Data shown above, we can get a better understanding about the individuals' information collected in the dataset.")
  st.markdown("The mean age for the participants is about 24 years old, the mean height of the participants is about 1.7 metres, and the mean weight of the participants is 86.59 kg.")
  st.markdown("### Missing Values")
  st.markdown("Null or NaN values.")

  dfnull = df.isnull().sum()/len(df)*100
  totalmiss = dfnull.sum().round(2)
  st.write("Percentage of total missing values:",totalmiss)
  st.write(dfnull)
  if totalmiss == 0.0:
    st.success("✅ We do not exprience any missing values which is the ideal outcome of our data. We can proceed with higher accuracy in our further prediction.")
  else:
    st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

  st.markdown("### Completeness")
  st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

  st.write("Total data length:", len(df))
  nonmissing = (df.notnull().sum().round(2))
  completeness= round(sum(nonmissing)/len(df),2)

  st.write("Completeness ratio:",completeness)
  st.write(nonmissing)
  if completeness >= 0.80:
    st.success("✅ We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")
  else:
    st.success("Poor data quality due to low completeness ratio (less than 0.85).")

elif app_mode == '02 Visualization':
  df=pd.read_csv("FINAL PROJECT.csv")

  varibles = st.sidebar.radio("Pick the varible",["PERSONAL HABITS","EATING HABITS","PHYSICAL ACTIVITY","FAMILY HISTORY"])

  if varibles == "PERSONAL HABITS":
    st.header("Personal Habits")
    st.subheader("SMOKE: Do you smoke?")
    crosstab = pd.crosstab(df['SMOKE'], df['OL'])
    fig, ax = plt.subplots()
    crosstab.plot(kind='bar',width=0.8, ax=ax)
    for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)
    st.write('For the "no" category of "SMOKE", the bars are quite tall, suggesting higher counts for each obesity level, with "Obesity Type I" showing the highest count at 345. The "yes" category has significantly lower counts for each obesity level, which might indicate that "SMOKE" is not a key influential variable for obesity level.')

    st.subheader("CALC: How often do you drink alcohol?")
    crosstab = pd.crosstab(df['CALC'], df['OL'])
    fig, ax = plt.subplots()
    crosstab.plot(kind='bar',width=0.8, ax=ax)
    for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)

    st.subheader("SCC: Do you monitor the calories you eat daily")
    crosstab = pd.crosstab(df['SCC'], df['OL'])
    fig, ax = plt.subplots()
    crosstab.plot(kind='bar',width=0.8, ax=ax)
    for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)

    st.subheader("MTRANS: Which transportation do you usually use?")
    crosstab = pd.crosstab(df['MTRANS'], df['OL'])
    fig, ax = plt.subplots()
    crosstab.plot(kind='bar',width=0.8, ax=ax)
    for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)

  elif varibles=='EATING HABITS':

    st.header("Eating Habits")
    st.subheader("CAEC: DO you eat any food between meals?")
    crosstab = pd.crosstab(df['CAEC'], df['OL'])
    fig, ax = plt.subplots()
    crosstab.plot(kind='bar',width=0.8, ax=ax)
    for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)

    st.subheader("FAVC : Do you eat high caloric food frequently??")
    crosstab = pd.crosstab(df['FAVC'], df['OL'])
    fig, ax = plt.subplots()
    crosstab.plot(kind='bar',width=0.8, ax=ax)
    for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)

    st.subheader("FCVC : Do you usually eat vegetables in your meals?")
    crosstab = pd.crosstab(df['FAVC'], df['OL'])
    fig, ax = plt.subplots()
    crosstab.plot(kind='bar',width=0.8, ax=ax)
    for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)

  elif varibles=='PHYSICAL ACTIVITY':

    df_ot = df[df["OL"] == 'Obesity_Type_I' ]
    df_ot2 = df[df["OL"] == 'Obesity_Type_II']
    df_ot3 = df[df["OL"] == 'Obesity_Type_III']

    df_ot_final = pd.concat([df_ot,df_ot2,df_ot3])      # data frem of Obesity_Type I, II, III
    df_ot_final.reset_index(drop=True, inplace = True)

    df_ow = df[df["OL"]=='Overweight_Level_I']
    df_ow2 = df[df["OL"]=='Overweight_Level_II']

    df_ow_final = pd.concat([df_ow,df_ow2])    # data frem of Over_weight_Type I, II
    df_ow_final.reset_index(drop=True, inplace = True)

    df_n = df[df["OL"]=='Normal_Weight']

    df_In = df[df["OL"]=='Insufficient_Weight']

    st.header('Physical Activity')
    st.subheader("FAF: How often do you have physical activity?")
    data_list = [df_ot_final, df_ow_final, df_n, df_In]
    data_name = ["obesity_type", "over_weight_type", "normal", "Insufficient_Weight"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))

    for i in range(2):
      sns.kdeplot(ax=axes[i,0], data=data_list[i], x="FAF", hue="OL", fill=True)
      axes[i, 0].set_title(f'{data_name[i]} vs FAF')

      sns.kdeplot(ax=axes[i,1], data=data_list[i+2], x="FAF", hue="OL", fill=True)
      axes[i, 1].set_title(f'{data_name[i+2]} vs FAF')

    fig.suptitle('Obesity_levels vs FAF')
    plt.tight_layout()
    st.pyplot(fig)

  elif varibles=='FAMILY HISTORY':

    st.header('Family History')
    st.subheader("FHWO: Family history with overweight")
    crosstab = pd.crosstab(df['FHWO'], df['OL'])
    fig, ax = plt.subplots()
    crosstab.plot(kind='bar',width=0.8, ax=ax)
    for p in ax.patches:
      ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)



if app_mode == '03 Prediction':
  image_2 = Image.open('image2.png')
  st.image(image_2, width=300)
  #Data Preprocessing
  df = pd.read_csv("FINAL PROJECT.csv")
  X = df.drop('OL', axis=1)
  y = df['OL']

  # Convert categorical columns using get_dummies (one-hot encoding)
  X = pd.get_dummies(X)
  #train test split
  X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)

  #normalize the features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  model_choice = st.sidebar.selectbox('Select to see:', ['KNN', 'Random Forest','Comparison Analysis'])
  if model_choice == 'KNN':

    ##The KNN Model
    knn = KNeighborsClassifier (n_neighbors = 3)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.title("Prediction - k-nearest neighbors Model:")
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")
   ##select box --
    option = st.selectbox(
      'What would you like to see❓',
      ('Confusion Matrix 📈', 'Predicted Results with Classification Report📑')
    )
    if option == 'Confusion Matrix 📈':
    ##KNN Confusion Matrix
      conf_matrix = confusion_matrix (y_test, y_pred)
      plt.figure(figsize=(10, 8))
      sns.heatmap(conf_matrix, annot=True, fmt="d")
      plt.xlabel('Predicted Labels')
      plt.ylabel('True Labels')
      plt.title('Confusion Matrix: KNN Model Prediction')
      st.pyplot(plt)
      st.markdown("#### The Labels from 0-6 indicates that:")
      st.markdown("0 = Insufficient Weight  1 = Normal Weight")
      st.markdown("2 = Obesity Type I  3 = Obesity Type II  4 = Obesity Type III")
      st.markdown("5 = Overweight Level I  6 = Overwight Level II")
      st.markdown("#### Explaination of the graph:")
      st.markdown(" The numbers in the matrix represent the counts of instances for each true-predicted label pair.")
      st.markdown(" The DIAGONAL from the top-left to the bottom-right shows the number of CORRECT predictions for each class.")
      st.markdown(" The NON-DIAGONAL numbers indicate the instances where the model made ERRORS, showing how many it made in this Model")
    elif option == 'Predicted Results with Classification Report📑':
      #KNN Results
      results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
      results.reset_index(drop=True, inplace=True)
      st.dataframe(results.head(10))
      #Report
      report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
      st.text('Classification Report:')
      st.table(report)

  elif model_choice == 'RandomForest':
    st.title("Prediction - Random Forest Classifier Model :")
    ##The Random Forest Classifier Model
    df = pd.read_csv('FINAL PROJECT.csv')
    # Identifying categorical columns (excluding the target variable 'OL')
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('OL')

    # Encoding
    preprocessor = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ], remainder='passthrough')

    # Pipeline with preprocessing and RandomForestClassifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    y = df['OL']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['OL']), y, test_size=0.2, random_state=42)

    # Fit the model on the training data
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")
    option = st.selectbox(
      'What would you like to see❓',
      ('Confusion Matrix 📈', 'Predicted Results with Classification Report📑')
    )
    if option == 'Confusion Matrix 📈':
      plt.figure(figsize=(10, 7))
      conf_matrix = confusion_matrix (y_test, y_pred)
      sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=pipeline.named_steps['classifier'].classes_, yticklabels=pipeline.named_steps['classifier'].classes_)
      plt.xlabel('Predicted')
      plt.ylabel('True')
      plt.title('Confusion Matrix')
      st.pyplot(plt)
    elif option == 'Predicted Results with Classification Report📑':
      #RandomForest Results
      results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
      results.reset_index(drop=True, inplace=True)
      st.dataframe(results.head(10))
      #Report
      report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
      st.text('Classification Report:')
      st.table(report)
  elif model_choice == 'Comparison Analysis':
    st.title('Comparison ⚖️')
    st.markdown('#### Confusion Matrices:')
    st.markdown ("KNN Model: High Accuracy for 'Obesity Type III'(Label 4) and 'Overweight Level II'(Label 6)")
    st.markdown ("Struggles with 'Normal weight' and 'Overweight Level I'")
    st.markdown ("RandomForestClassifier: Shows fewer misclassifications overall. Significantly in distinguishing 'Normal Weight and 'Overweight Level I'")
    st.markdown ('#### Classification Reports:')
    st.markdown ("KNN: The recall for 'Normal Weight' is particularly low (0.4677), indicating many instances of this class were misclassified.")
    st.markdown ("But the precision is high for Obesity prediction, especially 'Obesity Type III' with a perfect recall.")
    st.markdown ("RandomDorest Classifier: the recall for 'Normal Weight' is much improved to 0.9355")
    st.markdown ("#### OVERALL")
    st.markdown ("Accuracy: RandomForest Classifier increased from 0.8109 to 0.9433.")
    st.markdown ("F1-Score: The harmonic mean of precision and recall is higher in the RandomForest model, suggesting a better balance.")

if app_mode == '04 Explainable AI':
    st.title('Explainable AI: Shapash')
    df = pd.read_csv('FINAL PROJECT.csv')

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'FHWO', 'CAEC', 'MTRANS', 'OL']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    X = df[['CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE', 'CH2O', 'FHWO', 'FAF', 'TECH', 'CAEC', 'MTRANS']]
    y = df['OL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model training
    model = RandomForestRegressor(max_depth=5, random_state=42, n_estimators=12)
    model.fit(X_train, y_train)

    option = st.selectbox(
        'What would you like to see❓',
        ('Feature Importance', 'Feature Contribution', 'Local Explanation')
    )

    if option == 'Feature Importance':
        # Make predictions and format them as a DataFrame
        y_pred = pd.DataFrame(model.predict(X_test), columns=['pred'], index=X_test.index)
        xpl = SmartExplainer(model=model)  # Pass the model correctly
        xpl.compile(x=X_test, y_pred=y_pred)  # Use the correctly formatted predictions
        fig = xpl.plot.features_importance()
        st.write(fig)
    if option == 'Feature Contribution':
        feature_list = X_test.columns.tolist()
        selected_feature = st.selectbox('Select a feature for the contribution plot:', feature_list)
        y_pred = pd.DataFrame(model.predict(X_test), columns=['pred'], index=X_test.index)
        xpl = SmartExplainer(model=model)  # Pass the model correctly
        xpl.compile(x=X_test, y_pred=y_pred)  # Use the correctly formatted predictions
        fig = xpl.plot.contribution_plot(selected_feature)
        st.write(fig)
    if option == 'Local Explanation':
        y_pred = pd.DataFrame(model.predict(X_test), columns=['pred'], index=X_test.index)
        xpl = SmartExplainer(model=model)  # Pass the model correctly
        xpl.compile(x=X_test, y_pred=y_pred)  # Use the correctly formatted predictions
        fig = xpl.plot.local_plot(index=random.choice(X_test.index))
        st.write(fig)

if app_mode == '05 MLFlow Tracking':
  def load_data():
    df = pd.read_csv('FINAL PROJECT.csv')
    df['target'] = df['OL']
    return df

  # Preprocessing function to encode categorical variables
  def preprocess_features(df, feature_choices):
    categorical_features = df[feature_choices].select_dtypes(include=['object']).columns.tolist()
    numeric_features = df[feature_choices].select_dtypes(exclude=['object']).columns.tolist()

    # Create transformers for numeric and categorical data
    numeric_transformer = 'passthrough'  # No transformation needed for numeric data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')  # OneHot encode categorical data

    # Create a column transformer to apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

  # Available models and their problem types
  MODELS = {
    "classification": {
        "KNN": KNeighborsClassifier,
        "SVM": SVC,
        "Random Forest": RandomForestClassifier
    }
  }

  st.title("Model Experimentation with MLflow 🚀")

  # User selects the task type
  task_type = st.selectbox("Select the task type:", ["classification"])

  # Load data
  df = load_data()
  st.write(df)

  # Model and feature selection
  model_options = list(MODELS[task_type].keys())
  model_choice = st.selectbox("Choose a model ⚙️", model_options)
  feature_options = df.columns.drop('target').tolist()  # Adjust 'target' as necessary
  feature_choice = st.multiselect("Choose some features", feature_options)
  target_choice = st.selectbox("Select target column", df.columns)

  # Preprocess selected features
  preprocessor = preprocess_features(df, feature_choice)

  # MLflow tracking
  track_with_mlflow = st.checkbox("Track with MLflow?")

  # Model training and evaluation
  if st.button("Start training"):
    if track_with_mlflow:
        mlflow.set_experiment("Obesity_Prediction")
        mlflow.start_run()
        mlflow.log_param('model', model_choice)
        mlflow.log_param('features', feature_choice)
        mlflow.log_param('task', task_type)

    # Create a pipeline with preprocessing and model
    model = MODELS[task_type][model_choice]()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    X = df[feature_choice]
    y = df[target_choice]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pipeline.fit(X_train, y_train)

    # Evaluate the model
    preds_train = pipeline.predict(X_train)
    preds_test = pipeline.predict(X_test)
    if task_type == "classification":
        metric_train = f1_score(y_train, preds_train, average='micro')
        metric_test = f1_score(y_test, preds_test, average='micro')
    else:
        metric_train = r2_score(y_train, preds_train)
        metric_test = r2_score(y_test, preds_test)
    st.write("metric_train", round(metric_train, 3))
    st.write("metric_test", round(metric_test, 3))

    if track_with_mlflow:
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_metric("metric_train", metric_train)
        mlflow.log_metric("metric_test", metric_test)
        mlflow.end_run()

    with open('model.pkl', 'wb') as file:
      pickle.dump(pipeline, file)

  def download_file():
    file_path = 'model.pkl'  # Replace with the actual path to your model.pkl file
    with open(file_path, 'rb') as file:
      contents = file.read()
    b64 = base64.b64encode(contents).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="model.pkl">Download model.pkl file</a>'
    st.markdown(href, unsafe_allow_html=True)

  st.title("Download Model Example")
  st.write("Click the button below to download the model.pkl file.")
  if st.button("Download"):
    download_file()
