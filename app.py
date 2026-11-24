from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import plotly.express as px

st.set_page_config(page_title='Student Performance – SEC C', page_icon='🎓', layout='wide')
BASE = Path(__file__).resolve().parent
st.markdown("""<style>
[data-testid="stSidebar"] {background:#E8EEF5;}
[data-testid="stMetric"] {background:white;border:1px solid #DCE5EF;border-radius:14px;padding:16px;}
.block-container {padding-top:2rem;max-width:1450px;}
</style>""",unsafe_allow_html=True)
st.title('Student Performance & Academic Risk Dashboard')
st.caption('SEC C | Original Google Colab models • inference only')
if sklearn.__version__ != '1.6.1':
    st.error('Install scikit-learn==1.6.1 using requirements.txt before loading the model.')
    st.stop()
@st.cache_resource
def load_model():
    return joblib.load(BASE / 'models/student_model.pkl')
m = load_model()
res = pd.read_csv(BASE / 'data/student_predictions.csv', dtype={'student_id':str})

def original_graph(i, title):
    st.subheader(title)
    st.image(str(BASE / 'assets' / f'{i}.png'))
    st.caption('Original saved Google Colab output')
features = m['features']
nums = m['numeric_features']
maps = {'gender':{'Male':0,'Female':1}, 'parental_education':{'School':0,'Graduate':1,'Postgraduate':2}, 'extracurricular':{'No':0,'Yes':1}}

def predict(raw):
    missing = [c for c in features if c not in raw]
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    if raw.empty: raise ValueError('Dataset is empty.')
    x = raw[features].copy()
    for c in nums:
        x[c] = pd.to_numeric(x[c], errors='coerce')
        limit = 16 if c == 'study_hours' else 100
        if x[c].isna().any() or not x[c].between(0,limit).all(): raise ValueError(f'{c}: enter valid numbers from 0 to {limit} in every row.')
    for c, mapping in maps.items():
        x[c] = x[c].astype(str).str.strip().str.title().map(mapping)
        if x[c].isna().any(): raise ValueError(f'{c}: allowed values are {list(mapping)}')
    clean = m['imputer'].transform(m['scaler'].transform(x))
    p = m['logistic_model'].predict_proba(clean)[:, list(m['logistic_model'].classes_).index(1)]
    f = 1-p
    cluster = m['kmeans_model'].predict(m['cluster_scaler'].transform(x[m['cluster_columns']]))
    distances, indices = m['nearest_neighbors'].kneighbors(clean)
    out = pd.DataFrame({'student_id':raw['student_id'].astype(str).values if 'student_id' in raw else [f'NEW{i+1:04}' for i in range(len(raw))], 'predicted_final_marks':np.clip(m['linear_model'].predict(clean),0,100).round(2), 'pass_probability':p.round(4), 'fail_probability':f.round(4), 'pass_fail_prediction':np.where(p>=.5,'Pass','Fail'), 'academic_risk':np.select([f<.30,f<.60],['Low','Medium'],default='High'), 'segment':[m['cluster_names'][i] for i in cluster], 'knn_prediction':np.where(m['knn_model'].predict(clean)==1,'Pass','Fail'), 'similar_training_positions':['; '.join(str(i) for i in row) for row in indices], 'nearest_distance':distances[:,0].round(3)})
    return out

def display_results(out, key):
    a,b,c=st.columns(3)
    a.metric('Students',len(out)); b.metric('Mean predicted marks',f"{out.predicted_final_marks.mean():.2f}"); c.metric('High risk',int((out.academic_risk=='High').sum()))
    st.dataframe(out,use_container_width=True)
    a,b=st.columns(2)
    a.plotly_chart(px.histogram(out,x='predicted_final_marks',nbins=20,title='Predicted marks distribution'),key=key+'marks')
    b.plotly_chart(px.pie(out,names='pass_fail_prediction',hole=.55,title='Predicted pass / fail'),key=key+'pass')
    st.plotly_chart(px.scatter(out,x='student_id',y='pass_probability',color='academic_risk',title='Student-level pass probability'),key=key+'prob')
    a,b=st.columns(2)
    a.plotly_chart(px.histogram(out,x='academic_risk',color='academic_risk',title='Academic risk'),use_container_width=True,key=key+'risk')
    b.plotly_chart(px.histogram(out,x='segment',color='segment',title='Performance groups'),use_container_width=True,key=key+'group')
    st.caption('Similar training positions are zero-based positions in the saved neighbor model, not student IDs. Probabilities are model estimates, not guarantees.')
    st.download_button('Download predictions',out.to_csv(index=False),key+'.csv','text/csv',key=key+'download')

page=st.sidebar.radio('Navigation',['Overview','Preprocessing & Analysis','Model Performance','Single Student','Upload New Dataset','Risk & Segmentation','Feature Importance'])
if page=='Overview':
    a,b,c=st.columns(3); a.metric('Original records',610); b.metric('Clean records',len(res)); c.metric('Pass mark',m['pass_mark'])
    st.info('This dashboard loads the original Colab model. It never retrains on uploaded data.')
    st.dataframe(res,use_container_width=True)
    display_results(res,'overview')
    st.plotly_chart(px.scatter(res,x='actual_final_marks',y='predicted_final_marks',color='academic_risk',title='Original actual vs predicted marks'))
elif page=='Preprocessing & Analysis':
    st.subheader('Original Colab preprocessing')
    st.write('610 raw records → 600 cleaned records; 10 duplicates removed. Median filling, categorical encoding, IQR clipping, scaling and imputation were used in training.')
    st.dataframe(pd.DataFrame({'Feature':['attendance','study_hours','previous_marks','assignment_marks'],'Missing before':[15,15,16,15],'Missing after':[0,0,0,0]}))
    st.plotly_chart(px.histogram(res,x='actual_final_marks',title='Original final marks distribution'))
    st.caption('Original raw training CSV was not supplied. These feature plots are preserved directly from the executed notebook.')
    st.code((BASE/'assets/2.txt').read_text(),language=None)
    original_graph(4,'Correlation heatmap')
    a,b=st.columns(2)
    with a: original_graph(5,'Attendance vs final marks')
    with b: original_graph(6,'Study hours vs final marks')
    original_graph(7,'Previous marks vs final marks')
elif page=='Model Performance':
    st.subheader('Original Colab test metrics')
    st.dataframe(pd.DataFrame(list(m['model_metrics'].items()),columns=['Metric','Value']))
    st.write('5-fold Linear Regression CV R²: 0.708 | Logistic Regression CV accuracy: 0.808 | Best C: 1')
    st.caption('These are saved training-run metrics, not evaluation of new uploads. Original preprocessing was performed before splitting; CV scores may be optimistic.')
    st.plotly_chart(px.bar(x=['Logistic Regression','KNN'],y=[.842,.758],labels={'x':'Model','y':'Test accuracy'},title='Classification model comparison'))
    for i,t in [(9,'Linear Regression: actual vs predicted'),(10,'Logistic Regression: confusion matrix'),(11,'KNN: confusion matrix')]:
        original_graph(i,t)
        st.code((BASE/'assets'/f'{i}.txt').read_text(),language=None)
elif page=='Single Student':
    with st.form('single'):
        sid=st.text_input('Student ID','0022'); a,b=st.columns(2)
        values={c:(a if i%2==0 else b).number_input(c.replace('_',' ').title(),0.0,16.0 if c=='study_hours' else 100.0,value=float([56,10,56,25,45][i])) for i,c in enumerate(nums)}
        for c, mapping in maps.items(): values[c]=st.selectbox(c.replace('_',' ').title(),list(mapping),index=1 if c=='parental_education' else 0)
        submit=st.form_submit_button('Predict student')
    if submit:
        try: display_results(predict(pd.DataFrame([{'student_id':sid,**values}])),'single')
        except ValueError as e: st.error(str(e))
elif page=='Upload New Dataset':
    st.subheader('Upload unseen student records')
    st.write('Required columns: '+', '.join(features))
    st.caption('No final_marks required. Use the same units and categories as Colab. Invalid or missing inputs are rejected rather than silently guessed.')
    st.download_button('Download sample input CSV',(BASE/'data/new_student_dataset.csv').read_bytes(),'new_student_dataset.csv','text/csv')
    upload=st.file_uploader('Choose CSV',type=['csv'])
    if upload is not None:
        try:
            raw=pd.read_csv(upload,dtype={'student_id':str}); st.dataframe(raw.head(20))
            st.write('Uploaded rows:',len(raw),'| Duplicate rows:',int(raw.duplicated().sum()))
            st.dataframe(raw.isna().sum().rename('Missing values'))
            raw=raw.drop_duplicates().reset_index(drop=True)
            if st.button('Run prediction',type='primary'): st.session_state['bulk_result']=predict(raw); st.session_state['bulk_source']=upload.getvalue()
            if st.session_state.get('bulk_source')==upload.getvalue(): display_results(st.session_state['bulk_result'],'bulk')
        except (ValueError,TypeError) as e: st.error(str(e))
elif page=='Risk & Segmentation':
    display_results(res,'original')
    original_graph(12,'Original K-Means student segmentation')
    st.subheader('Priority review: highest estimated risk')
    st.dataframe(res.sort_values('pass_probability').head(20))
else:
    imp=pd.DataFrame({'Feature':['study_hours','previous_marks','internal_marks','assignment_marks','attendance','parental_education','gender','extracurricular'],'Importance':[.445,.418,.304,.074,.041,.030,.002,0]})
    st.plotly_chart(px.bar(imp.sort_values('Importance'),x='Importance',y='Feature',orientation='h',title='Original Colab permutation importance'))
    st.caption('Rounded notebook values; predictive association, not causal effects.')
    original_graph(13,'Original permutation importance')
