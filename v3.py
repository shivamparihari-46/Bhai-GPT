import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import os
import subprocess

username = os.environ["GITHUB_USERNAME"]
token = os.environ["GITHUB_TOKEN"]

repo_url = f"https://{username}:{token}@github.com/shivamparihari-46/Bhai-GPT.git"


if not os.path.exists("Bhai-GPT"):
    subprocess.run(["git", "clone", repo_url])
    print("Private repo cloned successfully!")
else:
    print("ℹRepo already exists, skipping clone.")

st.set_page_config(
    page_title="Bhai-GPT", 
    layout="wide",
    page_icon="./assets/icon.jpg"
    )


st.markdown("""
    <style>
        /* Table font size */
        .dataframe th, .dataframe td {
            font-size: 18px !important;
        }
        
        /* Make main content wider and centered */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
         
        /* General text */
        html, body, [class*="css"]  {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)



st.title("📊Bhai-GPT – Your Desi Data Scientist")
st.markdown(
    "<p style='font-size:28px; font-weight:600;'>Why do it yourself?? when Bhai-GPT is here!!</p>", 
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload your CSV file bhai, save 60% of your time and go on dates!", type=['csv'])

if uploaded_file is not None:
    x = pd.read_csv(uploaded_file)
    
    for col in x.select_dtypes(include='object').columns:
        x[col] = x[col].astype(str)

    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Bhai, kya dekhna chahta hai?", 
        [" Data Overview", 
         " Null Handling", 
         " Graph Visualization", 
         " Pairplot", 
         " Unique Values",
         "corelation-map",
         "ML Predictor",
         "SHAP Analysis",
         "Report"
         ]
    )
    
    if section == " Data Overview":
        st.success("Data received succesfully bhai!")
        st.markdown("<p style='font-weight:600;font-size:20px; '>Data dekh le:</p>",unsafe_allow_html=True)
        st.dataframe(x.astype(str))

        st.markdown("<p style='font-weight:600;font-size:20px; '> Data summary:</p>",unsafe_allow_html=True)
        st.write(x.describe(include='all'))
        st.write(f"Shape of your data: {x.shape[0]} rows × {x.shape[1]} columns")
        
    elif section == " Null Handling":
        st.markdown("<p style='font-weight:600;font-size:20px; '>Null values:</p>",unsafe_allow_html=True)
        st.write(x.isnull().sum())

        st.markdown("<p style='font-weight:600;font-size:20px; '>datatypes:</p>",unsafe_allow_html=True)
        st.write(x.dtypes)
                
        cols = st.multiselect("Select columns of which you want Null values to be replaced:", x.select_dtypes(include=['number']).columns)
        result = st.selectbox("Koi numerical null value replace karwana hai?", ['y', 'n'])
        
        if result == 'y' and len(cols) > 0:
            for i in cols:
                x[i] = x[i].fillna(x[i].mean())
                  

            st.success("Bhai ne selected columns ke nulls ko mean se bhar diya! ")
            st.write("Updated null values:")
            st.write(x[cols].isnull().sum())
        elif result == 'y' and len(cols) == 0:
            st.warning("Bhai, koi column to select kar le null fill ke liye")
        elif result=='n':
            st.write("thik bhai tu replace nhi karana chahta ")
        
        object_cols = x.select_dtypes(include=['object', 'category']).columns
        obj_2 = [col for col in object_cols if x[col].isnull().sum() > 0]
        if len(obj_2) > 0:
            st.markdown("#### Object type nulls detected")
            st.write(x[obj_2].isnull().sum()) 
            
            if st.button("Replace all object/categorical nulls with 'missing'",type='primary'):
                for col in obj_2:
                     x[col] = x[col].fillna("missing")
                     st.success("Bhai,Null values replaced successfully!!")
                     st.write(x[obj_2].isnull().sum())   
            
    elif section==" Unique Values":
        st.markdown("<p style='font-weight:600;font-size:20px; '>Unique values of columns:</p>",unsafe_allow_html=True)
        for i in x.columns:
            with st.expander(f" Unique values in **{i}**"):
             st.write(x[i].unique())
       
        
    elif section == " Graph Visualization":
    
        st.markdown("### Select the columns to be plotted or analyzed :")
        selected_cols = st.multiselect("Select 2 columns bhai", x.columns)

        if len(selected_cols) == 2:
            col1, col2 = selected_cols[0], selected_cols[1]

            st.success(f"Bhai visualize karega: {col1} vs {col2}")

            plot_type = st.selectbox("Kaunsa graph chahiye?", ["Scatterplot", "Boxplot", "Barplot", "Lineplot"])

            fig, ax = plt.subplots()
            if plot_type == "Scatterplot":
                sns.scatterplot(data=x, x=col1, y=col2, ax=ax)
            elif plot_type == "Boxplot":
                sns.boxplot(data=x, x=col1, y=col2, ax=ax)
            elif plot_type == "Barplot":
                sns.barplot(data=x, x=col1, y=col2, ax=ax,palette='cool')
            elif plot_type == "Lineplot":
                sns.lineplot(data=x, x=col1, y=col2, ax=ax)

            st.pyplot(fig)
        elif len(selected_cols) > 2:
            st.warning("Bhai, more than 2 columns selected ,could not plot")
        
    elif section == " Pairplot":    
        st.markdown("### pairplot chahiye? niche dekho") 
        selected_cols_2 = st.multiselect("Select the columns bhai", x.select_dtypes(include=['number']).columns)
        cat_cols = x.select_dtypes(include=['object', 'category']).columns

        target_column = st.selectbox(" Bhai, kaunsa column ko hue rakhe?",cat_cols)
        palette_options = ['Set1', 'coolwarm', 'husl', 'Blues', 'viridis', 'mako', 'rocket','colorblind']
        palette_choice = st.selectbox(" Bhai, kaunsa color palette chahiye?", palette_options)


        
        if len(selected_cols_2)==0:
            pass
        elif len(selected_cols_2)>=2:
            
            fig= sns.pairplot(data=x,vars=selected_cols_2,hue=target_column,palette=palette_choice)

            st.pyplot(fig.figure)
            
        else:
            st.warning("Bhai , minimum 2 columns to chaiye na pair plot ke liye.... aise nahi ban payega")
        
    elif section=="ML Predictor":
        
        from sklearn.utils.multiclass import type_of_target
        output=st.selectbox("Choose your output column ",x.columns)
        
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        for i in x.columns:
            if x[i].dtype == 'object':
                x[i] = x[i].fillna('missing')
            else:
                x[i] = pd.to_numeric(x[i], errors='coerce')
                x[i] = x[i].fillna(x[i].mean())
                
        
             
                
        ip=x.drop(columns=output)
        op=x[output]  
        
        for col in ip.columns:
            if ip[col].nunique() > 1000:
                ip = ip.drop(columns=[col])
                
                
        label_encoders = {}
        for i in ip.columns:
            if ip[i].dtype == "object":
                le = LabelEncoder()
                ip[i] = le.fit_transform(ip[i])
                label_encoders[i] = le
                
        if op.dtype == "object":
            le_target = LabelEncoder()
            op = le_target.fit_transform(op)
        else:
            le_target = None
            
        ip = ip.apply(pd.to_numeric, errors='coerce')
        ip = ip.fillna(0)    
            
        
        if pd.Series(op).nunique() <= 1:

            st.error("Bhai, output column mein sirf ek hi class hai ya data galat hai. Model training nahi ho sakti.")
            st.stop()

        if pd.Series(op).isnull().any():

            st.error("Bhai, output column mein null values hain. Unhe remove ya fill karo.")
            st.stop()

        p_type = type_of_target(op)

        st.write(f"automatically detected problem type: **{p_type}**")
        
        Train_size=st.slider("train_size",0.55,0.99,0.8)
        from sklearn.model_selection import train_test_split
        xtrain,xtest,ytrain,ytest=train_test_split(ip,op,train_size=Train_size)   
        
        st.write(f"Training Set shape: {xtrain.shape}")
        st.write(f"Testing Set shape: {xtest.shape}")
        
        from sklearn.preprocessing import StandardScaler
        standard=StandardScaler()
        xtrain = standard.fit_transform(xtrain)
        xtest = standard.transform(xtest)
        
        try:
    


            if p_type=='continuous':
                from sklearn.linear_model import LinearRegression
                lr=LinearRegression()
                lr.fit(xtrain,ytrain)
                pred=lr.predict(xtest)

                from sklearn.metrics import r2_score, mean_squared_error

                r2 = r2_score(ytest, pred)
                mse = mean_squared_error(ytest, pred)

                st.write(f"R² Score: {r2}")
                st.write(f"Mean Squared Error: {mse}")

                
                e=pd.DataFrame({'y-test':list(ytest),'prediction':list(pred)})
                st.markdown("<p style='font-weight:600;font-size:20px; '>Best fit line:</p>",unsafe_allow_html=True)
                
                fig=sns.lmplot(x='y-test',y='prediction',data=e)
                st.pyplot(fig.fig)
                
                best_model = {"model": "linear regression", "r2": r2, "mse": mse}
                st.session_state['best_model_name'] = "linear regression"
                st.session_state['best_model_obj'] = lr
                st.session_state['xtrain'] = xtrain
                st.session_state['ytrain'] = ytrain
                st.session_state['xtest'] = xtest
                st.session_state['ytest'] = ytest
                st.session_state['features'] = ip.columns.tolist()

                
            elif p_type=='binary':
                from sklearn.svm import SVC
                from sklearn.linear_model import LogisticRegression
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.naive_bayes import GaussianNB
                from sklearn.tree import DecisionTreeClassifier 
                from sklearn.ensemble import RandomForestClassifier 
                from xgboost import XGBClassifier 
                from sklearn.inspection import permutation_importance
                
                
            
                
                models={
                "svm": SVC(),
                "logistic regression": LogisticRegression(),
                "knn": KNeighborsClassifier(),
                "naive bayes": GaussianNB(),
                "desicion tree ": DecisionTreeClassifier(),
                "random forest": RandomForestClassifier(),
                "xgb":XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,eval_metric='mlogloss',random_state=42,use_label_encoder=False)}
                
                result=[]
                from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
                
                for name,model in models.items():
                    model.fit(xtrain,ytrain)
                    pred=model.predict(xtest)
                    acc=accuracy_score(ytest,pred)
                    cm=confusion_matrix(ytest,pred)
                    f1 = f1_score(ytest, pred, average='weighted')
                    
                
                    result.append({
                        'model':name,
                        'accuracy':round(acc,4),
                        'f1 score':round(f1,4)
                        
                    })   
                    

                result_df = pd.DataFrame(result)
                st.markdown("### model comparison report:")
                st.dataframe(result_df.astype(str))


                best_model = max(result, key=lambda x: x['accuracy'])
                best_model_name = best_model['model']
                best_model_obj = models[best_model_name]
                best_model_obj.fit(xtrain, ytrain)   

                st.session_state['best_model_name'] = best_model_name
                st.session_state['best_model_obj'] = best_model_obj
                st.success(f"Best model: {best_model['model'].upper()} with accuracy {best_model['accuracy']}")
                
                fig, ax = plt.subplots()
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='model', y='accuracy', data=result_df, palette='viridis', ax=ax)
                ax.set_title("Model Accuracy Comparison")
                st.pyplot(fig)
            
            
                model_name = st.selectbox("Select type of model:", list(models.keys()))
                x = models[model_name] 
                x.fit(xtrain,ytrain)
                with st.spinner("Feature importance calculate ho raha bhai... wait kar"):
                     result = permutation_importance(x, xtest, ytest, n_repeats=3, random_state=42)


                importance_df = pd.DataFrame({
                    'Feature': ip.columns,
                    'Importance': result.importances_mean
                }).sort_values(by='Importance', ascending=False)
                st.markdown("#### feature importances:")
                st.dataframe(importance_df)
                
                fig, ax = plt.subplots(figsize=(12,6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='coolwarm')
                st.pyplot(fig)
                
              
                st.session_state['xtrain'] = xtrain
                st.session_state['xtest'] = xtest
                st.session_state['ytrain'] = ytrain
                st.session_state['ytest'] = ytest
                st.session_state['features'] = ip.columns.tolist()

         
            
            elif p_type == "multiclass":
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

                clf = RandomForestClassifier()
                clf.fit(xtrain, ytrain)
                pred = clf.predict(xtest)

                acc = accuracy_score(ytest, pred)
                cm = confusion_matrix(ytest, pred)
                report = classification_report(ytest, pred, output_dict=True)
                st.success(f"Multiclass classification done with accuracy: {acc:.4f}")
                st.dataframe(pd.DataFrame(report).T)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                
                best_model = {"model": "random forest", "accuracy": acc}
                st.session_state['best_model_name'] = "random forest"
                st.session_state['best_model_obj'] = clf
                st.session_state['xtrain'] = xtrain
                st.session_state['ytrain'] = ytrain
                st.session_state['xtest'] = xtest
                st.session_state['ytest'] = ytest
                st.session_state['features'] = ip.columns.tolist()

                
                
            elif p_type in ['multilabel-indicator', 'multioutput', 'multiclass-multioutput']:
        
                st.warning("multi label and multi output and multiclass detected. Ye abhi supported nahi hai.")

            else:
                st.error(f"kuch unexpected type mila: {p_type}")

        except Exception as e:
            st.error(f"Bhai, kuch error aa gaya model training mein: {e}")
            st.stop()
        
        
    elif section == "SHAP Analysis":
        if "best_model_obj" not in st.session_state:
            st.warning("Bhai pehle ML Predictor section mein model train kar.")
        else:
            st.success(f"SHAP analysis for best model: {st.session_state['best_model_name'].upper()}")
            model = st.session_state['best_model_obj']
            X_train = st.session_state['xtrain']
            X_test = st.session_state['xtest']
            features = st.session_state['features']

            import shap

            try:
                
                model_name = st.session_state['best_model_name'].lower()
                X_train_df = pd.DataFrame(X_train, columns=features)
                X_test_df = pd.DataFrame(X_test, columns=features)

                if "forest" in model_name or "tree" in model_name or "xgb" in model_name:
                    explainer = shap.TreeExplainer(model)
                elif "linear" in model_name or "logistic" in model_name:
                    explainer = shap.LinearExplainer(model, X_train_df)
                else:
                    explainer = shap.Explainer(model, X_train_df)

            
                shap_values = explainer(X_test_df)
               
               

                st.markdown("### SHAP Summary Plot:")

                if isinstance(shap_values, list): 
                    class_idx = st.selectbox("Kaun si class explain karni hai?", range(len(shap_values)))
                    shap.summary_plot(shap_values[class_idx], X_test_df, feature_names=features, show=False)
                else:
                    shap.summary_plot(shap_values, X_test_df, feature_names=features, show=False)

                st.pyplot(plt.gcf()) 


            except Exception as e:
                st.error(f"Bhai, SHAP analysis mein error aaya: {e}")


    elif section=="corelation-map":
        
        n=x.select_dtypes(include='number')
        fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(n.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
    elif section == "Report":
        st.markdown("## Generate PDF Report of Your Analysis")
        
        
        if "best_model_obj" not in st.session_state:
            st.warning("Bhai, pehle ML Predictor section run karo tabhi report generate kar sakte ho.")
        else:
            
            report_lines = [
                f"ML Model Report",
                "===================",
                f"Best Model: {st.session_state['best_model_name'].upper()}",
            ]
            
            
            if hasattr(st.session_state['best_model_obj'], 'predict'):
                ytest = st.session_state['ytest']
                ypred = st.session_state['best_model_obj'].predict(st.session_state['xtest'])
                try:
                    from sklearn.metrics import r2_score, mean_squared_error
                    r2 = r2_score(ytest, ypred)
                    mse = mean_squared_error(ytest, ypred)
                    report_lines.append(f"R2 Score: {r2:.4f}")
                    report_lines.append(f"Mean Squared Error: {mse:.4f}")
                except:
                    pass

            
            try:
                from sklearn.metrics import accuracy_score, classification_report
                ytest = st.session_state['ytest']
                ypred = st.session_state['best_model_obj'].predict(st.session_state['xtest'])
                acc = accuracy_score(ytest, ypred)
                report_lines.append(f"Accuracy: {acc:.4f}")
                
                
                report_dict = classification_report(ytest, ypred, output_dict=True)
                report_df = pd.DataFrame(report_dict).T
                report_lines.append("\nClassification Report:")
                for idx, row in report_df.iterrows():
                    report_lines.append(f"{idx}: {row.to_dict()}")
            except:
                pass
            
            
            if "features" in st.session_state:
                report_lines.append(f"\nFeatures used ({len(st.session_state['features'])}): {', '.join(st.session_state['features'])}")
            
            
            from fpdf import FPDF
            import io
            
            def create_pdf(report_lines):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in report_lines:
                    pdf.multi_cell(0, 8, txt=str(line))

                
                pdf_bytes = pdf.output(dest="S").encode("latin1")
                pdf_buffer = io.BytesIO(pdf_bytes)

                return pdf_buffer

            
            pdf_file = create_pdf(report_lines)
            
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name="ML_Report.pdf",
                mime="application/pdf"
            )




