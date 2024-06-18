import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

categorical_features = ['gender',
                        'region',
                        'highest_education',
                        'imd_band',
                        'age_band',
                        'disability',

                        ]

numeric_features = ['num_of_prev_attempts',
                    'studied_credits',
                    'unique_sites',
                    'nan_vle_activity_indicator',
                    'sum_click',
                    'avg_clicks',
                    'dataplus_count',
                    'dualpane_count',
                    'externalquiz_count',
                    'folder_count',
                    'forumng_count',
                    'glossary_count',
                    'homepage_count',
                    'htmlactivity_count',
                    'oucollaborate_count',
                    'oucontent_count',
                    'ouelluminate_count',
                    'ouwiki_count',
                    'page_count',
                    'questionnaire_count',
                    'quiz_count',
                    'repeatactivity_count',
                    'resource_count',
                    'sharedsubpage_count',
                    'subpage_count',
                    'url_count',
                    'dataplus_sum_clicks',
                    'dualpane_sum_clicks',
                    'externalquiz_sum_clicks',
                    'folder_sum_clicks',
                    'forumng_sum_clicks',
                    'glossary_sum_clicks',
                    'homepage_sum_clicks',
                    'htmlactivity_sum_clicks',
                    'oucollaborate_sum_clicks',
                    'oucontent_sum_clicks',
                    'ouelluminate_sum_clicks',
                    'ouwiki_sum_clicks',
                    'page_sum_clicks',
                    'questionnaire_sum_clicks',
                    'quiz_sum_clicks',
                    'repeatactivity_sum_clicks',
                    'resource_sum_clicks',
                    'sharedsubpage_sum_clicks',
                    'subpage_sum_clicks',
                    'url_sum_clicks',
                    'dataplus_perc_clicks',
                    'dualpane_perc_clicks',
                    'externalquiz_perc_clicks',
                    'folder_perc_clicks',
                    'forumng_perc_clicks',
                    'glossary_perc_clicks',
                    'homepage_perc_clicks',
                    'htmlactivity_perc_clicks',
                    'oucollaborate_perc_clicks',
                    'oucontent_perc_clicks',
                    'ouelluminate_perc_clicks',
                    'ouwiki_perc_clicks',
                    'page_perc_clicks',
                    'questionnaire_perc_clicks',
                    'quiz_perc_clicks',
                    'repeatactivity_perc_clicks',
                    'resource_perc_clicks',
                    'sharedsubpage_perc_clicks',
                    'subpage_perc_clicks',
                    'url_perc_clicks',
                    'days_until_submission',
                    'submited_assessments_count',
                    'sum_passed_assessments',
                    'sum_failed_assessments',
                    'nan_assessments_indicator',
                    'current_mean_score',
                    'transferred_assessments_count',
                    'date_registration'
                    ]





def load_data():
    studentInfo = pd.read_csv('../data/studentInfo.csv')
    courses = pd.read_csv('../data/courses.csv')
    vle = pd.read_csv('../data/vle.csv')
    assessments = pd.read_csv('../data/assessments.csv')
    studentRegistration = pd.read_csv('../data/studentRegistration.csv')
    studentAssessment = pd.read_csv('../data/studentAssessment.csv')
    studentVle = pd.read_csv('../data/studentVle.csv')
    return studentInfo, courses, vle, assessments, studentRegistration, studentVle, studentAssessment


def find_unique_sites(studentVle, studentInfo, date_threshold):
    studentVleRestricted =studentVle.loc[studentVle['date']<date_threshold,:]
    unique_sites_per_student_module_pres = studentVleRestricted.groupby(['id_student','code_module','code_presentation']).agg({'id_site':'nunique'}).reset_index()
    unique_sites_per_student_module_pres.rename(columns={'id_site':'unique_sites'},inplace=True)
    if date_threshold == 300.0:
        print(f"Test: for date_threshold 3000 shape of dataframe unique_sites_per_student_module_presentation should be: (29228,4). \n Got:{unique_sites_per_student_module_pres.shape}")
    all_students_module_presentation_unique_sites = pd.merge(studentInfo,unique_sites_per_student_module_pres, on=['id_student','code_module','code_presentation'],how='left')
    all_students_module_presentation_unique_sites['nan_vle_activity_indicator'] = all_students_module_presentation_unique_sites['unique_sites'].isna().astype('int')
    return all_students_module_presentation_unique_sites[['id_student','code_module','code_presentation','unique_sites','nan_vle_activity_indicator']]

def find_sum_of_the_clicks(studentVle, studentInfo, date_threshold):
    studentVleRestricted = studentVle.loc[studentVle['date']<date_threshold,:]
    total_clicks_per_course_module_student = studentVleRestricted.groupby(['id_student','code_module','code_presentation'])['sum_click'].sum().reset_index()
    all_students_module_presentation_sum_clicks = pd.merge(studentInfo, total_clicks_per_course_module_student,on=['id_student', 'code_module','code_presentation'],how='left')
    return all_students_module_presentation_sum_clicks[['id_student','code_module','code_presentation','sum_click']]

def find_average_daily_clicks(studentVle, studentInfo,courses, date_threshold):
    result = find_sum_of_the_clicks(studentVle, studentInfo,date_threshold)
    final_res = pd.merge(result, courses, on = ['code_module','code_presentation'])
    final_res['date_threshold'] = date_threshold
    final_res['avg_clicks'] = np.where(final_res['date_threshold']<final_res['module_presentation_length'], final_res['sum_click']/final_res['date_threshold'], final_res['sum_click']/final_res['module_presentation_length'])
    return final_res[['id_student','code_module','code_presentation','avg_clicks']]



def find_activity_type_counts(studentVle,vle,studentInfo,date_threshold):
    studentVle_vle = pd.merge(studentVle, vle,on=['code_module', 'code_presentation','id_site'])
    studentVle_vle_restricted = studentVle_vle.loc[studentVle_vle['date']<=date_threshold,:]
    student_mod_pres_activity_type_count = studentVle_vle_restricted.groupby(['id_student', 'code_module', 'code_presentation', 'activity_type']).size().unstack(fill_value=0).reset_index()
    student_mod_pres_activity_type_count.columns.name = None
    all_students_activity_type_counts = pd.merge(studentInfo,student_mod_pres_activity_type_count,on=['id_student', 'code_module', 'code_presentation'],how='left')
    result = pd.concat([all_students_activity_type_counts.iloc[:,:3],all_students_activity_type_counts.iloc[:,12:]],axis=1)
    for col in result.columns.to_list()[3:]:
        result.rename(columns={str(col):str(col)+'_count'},inplace=True)
    return result


def find_activity_type_sum_of_clicks(studentVle,vle,studentInfo,date_threshold):
    studentVle_vle = pd.merge(studentVle, vle,on=['code_module', 'code_presentation','id_site'])
    studentVle_vle_restricted = studentVle_vle.loc[studentVle_vle['date']<=date_threshold,:]
    student_mod_pres_activity_type_count = studentVle_vle_restricted.groupby(['id_student', 'code_module', 'code_presentation', 'activity_type'])['sum_click'].sum().unstack(fill_value=0).reset_index()
    student_mod_pres_activity_type_count.columns.name = None # because name was 'activity_type'
    all_students_activity_type_counts = pd.merge(studentInfo,student_mod_pres_activity_type_count,on=['id_student', 'code_module', 'code_presentation'],how='left')
    result = pd.concat([all_students_activity_type_counts.iloc[:,:3],all_students_activity_type_counts.iloc[:,12:]],axis=1)
    for col in result.columns.to_list()[3:]:
        result.rename(columns={str(col):str(col)+'_sum_clicks'},inplace=True)
    return result


def find_percentage_of_clicks_per_activity(studentVle,vle,studentInfo, date_threshold):
    activity_counts_df = find_activity_type_sum_of_clicks(studentVle,vle,studentInfo,date_threshold)
    sum_of_clicks_df = find_sum_of_the_clicks(studentVle,studentInfo,date_threshold)
    result_dataframe = pd.merge(activity_counts_df,sum_of_clicks_df,on=['id_student','code_module','code_presentation'])
    result_dataframe.iloc[:,3:-1] = result_dataframe.iloc[:,3:-1].div(result_dataframe.iloc[:,-1],axis=0)
    for col in result_dataframe.columns.to_list()[3:-1]:
        new_col = col.split('_sum_clicks')[0]+'_perc_clicks'
        result_dataframe.rename(columns={str(col):new_col},inplace=True)
    return result_dataframe.iloc[:,0:-1]


def find_mean_days_until_submission(assessments, courses, studentAssessment,studentInfo,date_threshold):
    courses_assessments = pd.merge(assessments, courses,on=['code_module','code_presentation'])
    courses_assessments['date']=np.where((courses_assessments['date'].isna()) & (courses_assessments['assessment_type']=='Exam'),courses_assessments['module_presentation_length'],courses_assessments['date'])
    studentAssessment_assessments = pd.merge(studentAssessment, courses_assessments, on=['id_assessment'])
    studentAssessment_assessmentsRestricted = studentAssessment_assessments.loc[(studentAssessment_assessments['date_submitted']<=date_threshold),:].copy()
    studentAssessment_assessmentsRestricted['days_until_submission'] = studentAssessment_assessmentsRestricted['date'] - studentAssessment_assessmentsRestricted['date_submitted']
    result_dataframe = studentAssessment_assessmentsRestricted.groupby(['id_student','code_module','code_presentation'])['days_until_submission'].mean().reset_index()
    all_students_days_until_submission = pd.merge(studentInfo,result_dataframe,on=['id_student','code_module','code_presentation'],how='left')[['id_student','code_module','code_presentation','days_until_submission']]
    return all_students_days_until_submission


def find_submited_assessments_count(studentAssessment, assessments, courses, studentInfo,date_threshold):
    studentAssessment_Restricted = studentAssessment.loc[studentAssessment['date_submitted']<=date_threshold,:].copy()
    studentAssessment_assessments_courses_Restricted = pd.merge(studentAssessment_Restricted,pd.merge(assessments,courses,on=['code_module','code_presentation']),on=['id_assessment'])
    all_submited = studentAssessment_assessments_courses_Restricted.groupby(['id_student','code_module','code_presentation'])['id_assessment'].count().reset_index()
    all_submited.rename(columns={'id_assessment':'submited_assessments_count'},inplace=True)
    result_dataframe = pd.merge(studentInfo,all_submited,on=['id_student','code_module','code_presentation'],how='left')[['id_student','code_module','code_presentation','submited_assessments_count']]
    result_dataframe['submited_assessments_count'].fillna(0,inplace=True)
    return result_dataframe


def find_number_of_passed_and_failed_assesments(studentAssessment, assessments, courses, studentInfo, date_threshold):
    studentAssessment_assessments_courses = pd.merge(studentAssessment, pd.merge(assessments,courses,on=['code_module','code_presentation']),on=['id_assessment'])
    studentAssessment_assessments_courses_Restricted = studentAssessment_assessments_courses.loc[studentAssessment_assessments_courses['date_submitted']<=date_threshold,:].copy()
    studentAssessment_assessments_courses_Restricted['passed_indicator'] = studentAssessment_assessments_courses_Restricted['score'].apply(lambda score:0 if score<40.0 else 1)  # pass = 1, fail = 0
    studentAssessment_assessments_courses_Restricted['failed_indicator'] = studentAssessment_assessments_courses_Restricted['score'].apply(lambda score:1 if score<40.0 else 0)

    all_passed_assessments_students = studentAssessment_assessments_courses_Restricted.groupby(['id_student','code_module','code_presentation'])['passed_indicator'].sum().reset_index() # koliko su do sada položili assessmenta
    all_passed_assessments_students.rename(columns={'passed_indicator':'sum_passed_assessments'},inplace=True)

    all_failed_assessments_students = studentAssessment_assessments_courses_Restricted.groupby(['id_student','code_module','code_presentation'])['failed_indicator'].sum().reset_index()
    all_failed_assessments_students.rename(columns={'failed_indicator':'sum_failed_assessments'},inplace=True)
    final_df = pd.merge(all_passed_assessments_students,all_failed_assessments_students,on=['id_student','code_module','code_presentation'])
    # all_students has 6750 nan values; after creating nan indicator, distinction has been made; students with NaN sum_passed assessments and indicator nan assessments of 1. when we fill nans there will be no problem
    all_students = pd.merge(studentInfo, final_df,on=['id_student','code_module','code_presentation'],how='left')[['id_student','code_module','code_presentation','sum_passed_assessments','sum_failed_assessments']]
    all_students['nan_assessments_indicator'] = all_students['sum_passed_assessments'].isna().astype('int')
    return all_students


def find_current_mean_score_and_transferred_assessments_count(studentInfo,studentAssessment,date_threshold):
    studentAssessmentRestricted = studentAssessment.loc[studentAssessment['date_submitted']<=date_threshold,:]
    meanScore = pd.merge(studentInfo, studentAssessmentRestricted,on=['id_student'],how='left').groupby(['id_student','code_module','code_presentation'])['score'].mean().reset_index()
    meanScore.rename(columns={'score':'current_mean_score'},inplace=True)
    isBankedCount = pd.merge(studentInfo,studentAssessmentRestricted,on=['id_student'],how='left').groupby(['id_student','code_module','code_presentation'])['is_banked'].count().reset_index()
    isBankedCount.rename(columns={'is_banked':'transferred_assessments_count'},inplace=True)
    result_df = pd.merge(meanScore,isBankedCount,on=['id_student','code_module','code_presentation'])
    return result_df


def createData(date_threshold):

    # creating dataset considering the day boundary
    studentInfo, courses, vle, assessments, studentRegistration, studentVle, studentAssessment  = load_data()
    unique_sites_df = find_unique_sites(studentVle, studentInfo, date_threshold)
    sum_of_clicks_df = find_sum_of_the_clicks(studentVle, studentInfo, date_threshold)
    avg_daily_clicks_df = find_average_daily_clicks(studentVle, studentInfo,courses, date_threshold)
    activity_type_counts_df = find_activity_type_counts(studentVle,vle,studentInfo,date_threshold)
    sum_of_clicks_per_activity_df = find_activity_type_sum_of_clicks(studentVle,vle,studentInfo,date_threshold)
    perc_of_clicks_per_activity_df = find_percentage_of_clicks_per_activity(studentVle,vle,studentInfo,date_threshold)
    mean_days_till_submission_df = find_mean_days_until_submission(assessments, courses, studentAssessment,studentInfo,date_threshold)
    submited_assessments_count_df = find_submited_assessments_count(studentAssessment, assessments, courses, studentInfo,date_threshold)
    passed_failed_df = find_number_of_passed_and_failed_assesments(studentAssessment, assessments, courses, studentInfo,date_threshold)
    mean_score_transferred_assessments_df = find_current_mean_score_and_transferred_assessments_count(studentInfo,studentAssessment,date_threshold)

    dataframes = [studentInfo, unique_sites_df, sum_of_clicks_df,avg_daily_clicks_df, activity_type_counts_df,sum_of_clicks_per_activity_df, perc_of_clicks_per_activity_df,mean_days_till_submission_df, submited_assessments_count_df,passed_failed_df,mean_score_transferred_assessments_df]
    merged_dataframe = dataframes[0]
    for df in dataframes[1:]:
        merged_dataframe = pd.merge(merged_dataframe,df,on=['id_student','code_module','code_presentation'])

    merged_dataframe_filtered = merged_dataframe[merged_dataframe['final_result']!='Fail']
    merged_filtered_registration = pd.merge(merged_dataframe_filtered,studentRegistration,on=['id_student','code_module','code_presentation'])
    merged_filtered_registration = merged_filtered_registration[merged_filtered_registration['date_registration']<date_threshold]
    merged_filtered_registration.drop(columns=['date_unregistration'],inplace=True)

    return merged_filtered_registration



def create_preprocessor(data): #,numeric_features=None,categorical_features=None

    global numeric_features
    global categorical_features
    """
    :param data: pd.DataFrame dataset
    :param numeric_features: all possible numeric features columns
    :param categorical_features: all possible categorical features columns
    :return: preprocessor, instance of ColumnTransformer
    """
    avaiabile_columns = data.columns.to_list()
    avaiabile_numeric_features = [feature for feature in avaiabile_columns if feature in numeric_features] # ako dostupna kolona feature postoji u svim mogućim num kolonama, onda potpada pod dostupne numericke kolone
    avaiabile_categorical_features = [feature for feature in avaiabile_columns if feature in categorical_features] # ako feature postoji u svim kategoričkim atributima, onda je jedan od dostupnih kategoričkih


    # 1. numeric transformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='constant',fill_value=0)),
        ('scaler',StandardScaler())
    ])

    # 2. categorical transformer:
    categorical_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoding',OneHotEncoder(handle_unknown='ignore')),
        ('scaler',StandardScaler(with_mean=False))
    ])

    #3. creating column_transformer with transformers list
    transformers = []
    transformers.append(('num',numeric_transformer,avaiabile_numeric_features))
    transformers.append(('cat',categorical_transformer,avaiabile_categorical_features))
    preprocessor = ColumnTransformer(transformers = transformers, remainder='drop')
    return preprocessor

def predict_labels(model,X_test_scaled,y_test):
    y_predicted = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test,y_predicted)
    precision = precision_score(y_test,y_predicted,labels=[1], average='weighted') # weighted, micro , average='weighted'
    recall = recall_score(y_test,y_predicted,labels=[1],average='weighted')
    fscore = f1_score(y_test,y_predicted,labels=[1],average='weighted')
    print(f"Accuracy:{accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {fscore}")



