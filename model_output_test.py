import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import lightgbm

BIGGER_DATA = False
if BIGGER_DATA:
    train_file = 'algebra_2005_2006_train.txt'
    test_file = 'algebra_2005_2006_test.txt'
else:
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'


def prepare():
    #cols = ['Personal CFAR', 'Problem CFAR','Anon Student Id','Problem Unit', 'Problem Section', 'Step Name', 'Problem Name', 'Problem View', 'Correct First Attempt','KC_length', 'KC_num','Unit CFAR', 'Section CFAR']
    cols = ['Personal CFAR', 'Problem CFAR', 'Anon Student Id', 'Problem Unit', 'Problem Section',
            'Step Name', 'Problem Name', 'Problem View', 'Correct First Attempt', 'KC_num', 'Step CFAR', 'KC CFAR']
    traindata = pd.read_csv(train_file, sep='\t')

    student_correct_rate = {}
    # CFAR Calculation
    for student, group in traindata.groupby(['Anon Student Id']):
        student_correct_rate[student] = (len(
            group[group['Correct First Attempt'] == 1]), len(group['Correct First Attempt']))
    traindata['Personal CFAR'] = traindata['Anon Student Id'].apply(
        lambda x: student_correct_rate[x][0])
    mean_SCFAR = np.mean(
        list(map(lambda x: x[0], list(student_correct_rate.values()))))
    problem_correct_rate = {}
    for problem, group in traindata.groupby(['Problem Name']):
        problem_correct_rate[problem] = 1.0 * len(
            group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])
    traindata['Problem CFAR'] = traindata['Problem Name'].apply(
        lambda x: problem_correct_rate[x])
    mean_PCFAR = np.mean(list(problem_correct_rate.values()))
    # Seperate
    traindata['Problem Unit'] = traindata['Problem Hierarchy'].str.split(
        ',', 1).str[0]
    traindata['Problem Section'] = traindata['Problem Hierarchy'].str.split(
        ',', 1).str[1]

    '''
    unit_correct_rate = {}
    for unit, group in traindata.groupby(['Problem Unit']):
        unit_correct_rate[unit] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])
    traindata['Unit CFAR'] = traindata['Problem Unit'].apply(lambda x: unit_correct_rate[x])
    mean_UCFAR = np.mean(list(unit_correct_rate.values()))
    
    section_correct_rate = {}
    for section, group in traindata.groupby(['Problem Section']):
        section_correct_rate[section] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])
    traindata['Section CFAR'] = traindata['Problem Section'].apply(lambda x: section_correct_rate[x])
    mean_SCFAR = np.mean(list(section_correct_rate.values()))
    '''

    step_correct_rate = {}
    for step, group in traindata.groupby(['Step Name']):
        step_correct_rate[step] = 1.0 * len(
            group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])
    traindata['Step CFAR'] = traindata['Step Name'].apply(
        lambda x: step_correct_rate[x])
    mean_STCFAR = np.mean(list(step_correct_rate.values()))

    KC_correct_rate = {}
    for KC, group in traindata.groupby(['KC(Default)']):
        if not pd.isnull(KC):  # KC != 'nan':
            KC_correct_rate[KC] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(
                group['Correct First Attempt'])
    mean_KCFAR = np.mean(list(KC_correct_rate.values()))
    traindata['KC CFAR'] = traindata['KC(Default)'].apply(
        lambda x: KC_correct_rate[x] if not pd.isnull(x) else mean_KCFAR)
    # print(mean_KCFAR)

    #traindata['KC_length'] = traindata['KC(Default)'].astype("str").apply(lambda x: len(x))
    traindata['KC_num'] = traindata['KC(Default)'].astype("str").apply(
        lambda x: 0 if x == 'nan' else (x.count('~~') + 1))
    # Seperate
    # train_x
    train_x = traindata[cols].copy()
    train_x['Opportunity(Mean)'] = traindata['Opportunity(Default)'].astype(
        "str").apply(lambda x: np.mean(list(map(int, x.replace('nan', '0').split('~~')))))
    train_x['Opportunity(Min)'] = traindata['Opportunity(Default)'].astype(
        "str").apply(lambda x: min(list(map(int, x.replace('nan', '0').split('~~')))))

    # Test
    testdata = pd.read_csv(test_file, sep='\t')
    testdata['Problem Unit'] = testdata['Problem Hierarchy'].str.split(
        ',', 1).str[0]
    testdata['Problem Section'] = testdata['Problem Hierarchy'].str.split(
        ',', 1).str[1]
    testdata['Personal CFAR'] = testdata['Anon Student Id'].apply(
        lambda x: student_correct_rate[x][0] if x in student_correct_rate.keys() else mean_SCFAR)
    testdata['Problem CFAR'] = testdata['Problem Name'].apply(
        lambda x: problem_correct_rate[x] if x in problem_correct_rate.keys() else mean_PCFAR)
    # Add
    #testdata['Unit CFAR'] = testdata['Problem Unit'].apply(lambda x: unit_correct_rate[x] if x in unit_correct_rate.keys() else mean_UCFAR)
    #testdata['Section CFAR'] = testdata['Problem Section'].apply(lambda x: section_correct_rate[x] if x in section_correct_rate.keys() else mean_SCFAR)
    testdata['Step CFAR'] = testdata['Step Name'].apply(
        lambda x: step_correct_rate[x] if x in step_correct_rate.keys() else mean_STCFAR)
    testdata['KC CFAR'] = testdata['KC(Default)'].apply(
        lambda x: KC_correct_rate[x] if x in KC_correct_rate.keys() else mean_KCFAR)
    #testdata['KC_length'] = testdata['KC(Default)'].astype("str").apply(lambda x: len(x))
    testdata['KC_num'] = testdata['KC(Default)'].astype("str").apply(
        lambda x: 0 if x == 'nan' else (x.count('~~') + 1))

    ####
    # test_x
    test_x = testdata[cols].copy()
    test_x['Opportunity(Mean)'] = testdata['Opportunity(Default)'].astype("str").apply(
        lambda x: np.mean(list(map(int, x.replace('nan', '0').split('~~')))))
    test_x['Opportunity(Min)'] = testdata['Opportunity(Default)'].astype(
        "str").apply(lambda x: min(list(map(int, x.replace('nan', '0').split('~~')))))

    # naive encoding
    sids = list(set(train_x['Anon Student Id']).union(
        set(test_x['Anon Student Id'])))
    sid_dict = {}
    for index, sid in enumerate(sids):
        sid_dict[sid] = index
    train_x['Anon Student Id'] = train_x['Anon Student Id'].apply(
        lambda x: sid_dict[x])
    test_x['Anon Student Id'] = test_x['Anon Student Id'].apply(
        lambda x: sid_dict[x])

    # naive encoding
    names = list(set(train_x['Problem Name']).union(
        set(test_x['Problem Name'])))
    names_dict = {}
    for index, name in enumerate(names):
        names_dict[name] = index
    train_x['Problem Name'] = train_x['Problem Name'].apply(
        lambda x: names_dict[x])
    test_x['Problem Name'] = test_x['Problem Name'].apply(
        lambda x: names_dict[x])

    # naive encoding
    units = list(set(train_x['Problem Unit']).union(
        set(test_x['Problem Unit'])))
    units_dict = {}
    for index, hierarchy in enumerate(units):
        units_dict[hierarchy] = index
    train_x['Problem Unit'] = train_x['Problem Unit'].apply(
        lambda x: units_dict[x])
    test_x['Problem Unit'] = test_x['Problem Unit'].apply(
        lambda x: units_dict[x])

    # naive encoding
    sections = list(set(train_x['Problem Section']).union(
        set(test_x['Problem Section'])))
    sections_dict = {}
    for index, hierarchy in enumerate(sections):
        sections_dict[hierarchy] = index
    train_x['Problem Section'] = train_x['Problem Section'].apply(
        lambda x: sections_dict[x])
    test_x['Problem Section'] = test_x['Problem Section'].apply(
        lambda x: sections_dict[x])

    # naive encoding
    sname = list(set(train_x['Step Name']).union(set(test_x['Step Name'])))
    sname_dict = {}
    for index, name in enumerate(sname):
        sname_dict[name] = index
    train_x['Step Name'] = train_x['Step Name'].apply(lambda x: sname_dict[x])
    test_x['Step Name'] = test_x['Step Name'].apply(lambda x: sname_dict[x])
    '''
    # # one hot encoding
    one_column = ['Anon Student Id','Problem Unit', 'Problem Section', 'Step Name', 'Problem Name']
    train_one = pd.get_dummies(train_x, columns = one_column, dummy_na=True)
    print(train_one)
    #test_one = pd.get_dummies(test_x[one_column], dummy_na=True, prefix=['col1', 'col2', 'col3', 'col4', 'col5'])

    another_column = ['Personal CFAR', 'Problem CFAR', 'Problem View', 'Correct First Attempt', 'KC_num', 'Step CFAR', 'KC CFAR']
    train_x = train_x[another_column].join(train_one)
    #test_x = test_x[another_column].join(test_one)
    '''
    train_x.to_csv('train_pre.csv', sep='\t', index=False)
    test_x.to_csv('test_pre.csv', sep='\t', index=False)


def train():
    if not BIGGER_DATA:
        train_df = pd.read_csv('train_pre.csv', sep='\t')
        test_df = pd.read_csv('test_pre.csv', sep='\t')
        
        X = train_df.dropna()
        y = np.array(X['Correct First Attempt']).astype(int).ravel()
        del X['Correct First Attempt']
        XX = test_df.dropna()
        yy = np.array(XX['Correct First Attempt']).astype(int).ravel()
        del XX['Correct First Attempt']
    else:
        train_df = pd.read_csv('train_pre.csv', sep='\t')
        X = train_df.dropna()
        y = np.array(X['Correct First Attempt']).astype(int).ravel()
        del X['Correct First Attempt']
        X, XX, y, yy = train_test_split(X,
                                        y,
                                        test_size=0.3,
                                        random_state=33)
    
    # 1. Decision Tree
    model = tree.DecisionTreeClassifier()
    model = model.fit(X, y)
    y_pred = model.predict(XX).astype(float)
    print ('Basic DecisionTree', np.sqrt(mean_squared_error(y_pred, yy)))

    # 2. Classifier
    clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=.01,
                                 n_jobs=4, random_state=None, verbose=0)
    clf.fit(X, y)
    y_pred = clf.predict_proba(XX)[:, 1]
    print ('RandomForest', np.sqrt(mean_squared_error(y_pred, yy)))

    ## 3. Adaboost
    clf = AdaBoostRegressor(base_estimator=None, n_estimators=50,
                            learning_rate=1.0, loss='exponential',
                            random_state=None)
    clf.fit(X, y)
    y_pred = clf.predict(XX)
    # print clf.best_estimator_
    print ('AdaBoost', np.sqrt(mean_squared_error(y_pred, yy)))
    
    ## 4. XGBoost
    '''
    param_dist = {
        'n_estimators': range(80, 200, 4),
        'max_depth': range(2, 15, 1),
        'learning_rate': np.linspace(0.01, 2, 20),
        'subsample': np.linspace(0.7, 0.9, 20),
        'colsample_bytree': np.linspace(0.5, 0.98, 10),
        'min_child_weight': range(1, 9, 1)
    }
    clf = GridSearchCV(estimator=XGBClassifier(learning_rate=0.2, n_estimators=140, max_depth=5,
                                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, scale_pos_weight=1),
                        param_grid=param_dist, scoring='neg_mean_squared_error', verbose=3, n_jobs=-1, iid=False, cv=5)
    '''
    
    clf = XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1, n_estimators=150, nthread=-1,
                        silent=0, gamma=0, min_child_weight=1, missing=0, subsample=.8, seed=33)
    clf.fit(X, y)
    y_pred = clf.predict_proba(XX)[:, 1]
    #print (clf.best_estimator_)
    print ('XGBClassifier', np.sqrt(mean_squared_error(y_pred, yy)))

    ##5. Lightgbm

    clf = lightgbm.LGBMClassifier(boosting_type = 'gbdt', objective='binary', max_depth=4, num_leaves = 31, learning_rate=0.1, n_estimators=2150, n_jobs=-1,
                        silent=0, min_child_weight=1, seed=33, subsample=0.85, subsample_freq = 1, boost_from_average = False, reg_lambda = 0.12, verbose = -1)
    
    '''
    param_dist = {
        'n_estimators': list(map(int,np.linspace(50, 4000, 4))),
        'max_depth': range(2, 15, 5),
        'num_leaves': range(31,63,20),
        'learning_rate': np.linspace(0.01, 2, 2),
        'subsample': np.linspace(0.7, 0.9, 2),
        'min_child_weight': range(1, 9, 5),
        'reg_lambda': np.linspace(0, 0.5, 2)
    }
    clf = GridSearchCV(estimator=lightgbm.LGBMClassifier(boosting_type = 'gbdt',seed=33, objective='binary', n_jobs=-1, subsample_freq = 1, silent=0, verbose = 0, boost_from_average = False, device_type = 'gpu'), param_grid=param_dist, scoring='neg_mean_squared_error', verbose=3, n_jobs=-1, iid=False, cv=5)
    '''
    
    clf.fit(X, y)
    y_pred = clf.predict_proba(XX)[:, 1]
    #print (clf.best_estimator_)
    print ('LGBMClassifier', np.sqrt(mean_squared_error(y_pred, yy)))

    ## 6. Gradient Decision Tree
    ### GBDT(Gradient Boosting Decision Tree) Classifier
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(X, y)
    y_pred = clf.predict_proba(XX)[:, 1]
    #print (clf.best_estimator_)
    print ('GBDTClassifier', np.sqrt(mean_squared_error(y_pred, yy)))

    ## 7.Logistic Regression
    clf = LogisticRegression(penalty='l2')
    clf.fit(X, y)
    y_pred = clf.predict_proba(XX)[:, 1]
    #print (clf.best_estimator_)
    print ('LogicalClassifier', np.sqrt(mean_squared_error(y_pred, yy)))
    
    # 8. Vote
    clf = VotingClassifier(estimators = [('rf', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=.01,\
    n_jobs=4, random_state=None, verbose=0)),('lgbm', lightgbm.LGBMClassifier(boosting_type = 'gbdt', objective='binary', max_depth=4, num_leaves = 31,\
            learning_rate=0.1, n_estimators=2150, n_jobs=-1,silent=0, min_child_weight=1, seed=33, subsample=0.85, subsample_freq = 1, boost_from_average = False, reg_lambda = 0.12, \
                verbose = -1))], voting = 'soft', weights=[1,1.5]) 
    clf.fit(X, y)
    y_pred = clf.predict_proba(XX)[:, 1]
    #print (clf.best_estimator_)
    print ('VoteClassifier', np.sqrt(mean_squared_error(y_pred, yy)))


def export(method="Vote"):
    train_df = pd.read_csv('train_pre.csv', sep='\t')
    test_df = pd.read_csv('test_pre.csv', sep='\t')

    X = train_df.dropna()
    y = np.array(X['Correct First Attempt']).astype(int).ravel()
    del X['Correct First Attempt']
    XX = test_df
    yy = np.array(XX['Correct First Attempt']).astype(float).ravel()
    del XX['Correct First Attempt']

    if method == 'DecisionTree':
        model = tree.DecisionTreeClassifier()
        model = model.fit(X, y)
        y_pred = model.predict(XX).astype(float)
    elif method == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=.01,
                                     n_jobs=4, random_state=None, verbose=0)
        clf.fit(X, y)
        y_pred = clf.predict_proba(XX)[:, 1]
    elif method == 'Adaboost':
        clf = AdaBoostRegressor(base_estimator=None, n_estimators=50,
                                learning_rate=1.0, loss='exponential',
                                random_state=None)
        clf.fit(X, y)
        y_pred = clf.predict(XX)
    elif method == 'XGBoost':
        clf = XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1, n_estimators=150, nthread=-1,
                            silent=0, gamma=0, min_child_weight=1, missing=0, subsample=.8, seed=33)
        clf.fit(X, y)
        y_pred = clf.predict_proba(XX)[:, 1]
    elif method == 'Lightgbm':
        clf = lightgbm.LGBMClassifier(boosting_type = 'gbdt', objective='binary', max_depth=4, num_leaves = 31, learning_rate=0.1, n_estimators=2150, n_jobs=-1,
                            silent=0, min_child_weight=1, seed=33, subsample=0.85, subsample_freq = 1, boost_from_average = False, reg_lambda = 0.12, verbose = -1)
        clf.fit(X, y)
        y_pred = clf.predict_proba(XX)[:, 1]
    elif method == 'GBDT':
        clf = GradientBoostingClassifier(n_estimators=200)
        clf.fit(X, y)
        y_pred = clf.predict_proba(XX)[:, 1]
    elif method == 'BasicLogistic':
        clf = LogisticRegression(penalty='l2')
        clf.fit(X, y)
        y_pred = clf.predict_proba(XX)[:, 1]
    elif method == 'Vote':
        print('Vote')
        clf = VotingClassifier(estimators = [('rf', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=.01,\
        n_jobs=4, random_state=None, verbose=0)),('lgbm', lightgbm.LGBMClassifier(boosting_type = 'gbdt', objective='binary', max_depth=4, num_leaves = 31,\
                learning_rate=0.1, n_estimators=2150, n_jobs=-1,silent=0, min_child_weight=1, seed=33, subsample=0.85, subsample_freq = 1, boost_from_average = False, reg_lambda = 0.12, \
                    verbose = -1))], voting = 'soft', weights=[1,1.5])
        clf.fit(X, y)
        y_pred = clf.predict_proba(XX)[:, 1]

    for index, val in enumerate(yy):
        if np.isnan(val):
            yy[index] = y_pred[index]
    test_res = pd.read_csv(test_file, sep='\t')
    test_res['Correct First Attempt'] = yy
    test_res.to_csv('test.csv', sep='\t', index=False)

prepare()
train()
export()
