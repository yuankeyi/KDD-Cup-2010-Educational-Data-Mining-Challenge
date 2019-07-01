from pyspark.sql.types import StringType
from pyspark.sql.functions import mean, col, udf
from pyspark import SQLContext, SparkContext
import numpy as np
import pandas

BIGGER_DATA = False
if BIGGER_DATA:
    train_file = 'algebra_2005_2006_train.txt'
    test_file = 'algebra_2005_2006_test.txt'
else:
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'

sc = SparkContext('local')
sqlContext = SQLContext(sc)

traindata = sqlContext.read.csv(train_file, sep='\t', header=True)
testdata = sqlContext.read.csv(test_file, sep='\t', header=True)


@udf
def get_unit(s):
    return s.split(',', 1)[0]


@udf
def get_section(s):
    return s.split(',', 1)[1]


@udf
def get_KCnum(s):
    return (s.count('~~')+1) if s else 0


@udf
def get_mean_opportunity(s):
    if not s:
        return 0
    tmp = list(map(int, s.split('~~')))
    return 1.0*sum(tmp)/len(tmp)


@udf
def get_min_opportunity(s):
    return min(list(map(int, s.split('~~')))) if s else 0


def naive_encoding(column):
    global traindata, testdata
    sid_dict = {}
    sids = [i[column]
            for i in traindata.union(testdata).select(column).distinct().collect()]
    for index, sid in enumerate(sids):
        sid_dict[sid] = index

    @udf
    def encoding(s):
        return sid_dict[s]

    traindata = traindata.withColumn(
        column, encoding(traindata[column]))
    testdata = testdata.withColumn(
        column, encoding(testdata[column]))


def drop(column):
    global traindata, testdata
    testdata = testdata.drop(column)
    traindata = traindata.drop(column)


def prepare():
    global traindata, testdata
    correct = traindata.filter(traindata['Correct First Attempt'] == '1')

    # Personal CFAR
    student_group = traindata.groupby('Anon Student Id').count()
    student_correct_group = correct.groupby('Anon Student Id').count()
    student_correct_rate = student_correct_group.join(student_group, student_group['Anon Student Id'] == student_correct_group['Anon Student Id']).drop(
        student_group['Anon Student Id']).select('Anon Student Id', (student_correct_group['count'] / student_group['count']).alias('Personal CFAR'))
    student_mean_CFAR = student_correct_rate.select(
        mean(col('Personal CFAR')).alias('mean')).collect()[0]['mean']
    traindata = traindata.join(student_correct_rate, student_correct_rate['Anon Student Id'] == traindata['Anon Student Id']).drop(
        student_correct_rate['Anon Student Id'])
    testdata = testdata.join(student_correct_rate, student_correct_rate['Anon Student Id'] == testdata['Anon Student Id']).drop(
        student_correct_rate['Anon Student Id'])
    testdata.na.fill(student_mean_CFAR, 'Personal CFAR')

    # Problem CFAR
    problem_group = traindata.groupby('Problem Name').count()
    problem_correct_group = correct.groupby('Problem Name').count()
    problem_correct_rate = problem_correct_group.join(problem_group, problem_group['Problem Name'] == problem_correct_group['Problem Name']).drop(
        problem_group['Problem Name']).select('Problem Name', (problem_correct_group['count'] / problem_group['count']).alias('Problem CFAR'))
    problem_mean_CFAR = problem_correct_rate.select(
        mean(col('Problem CFAR')).alias('mean')).collect()[0]['mean']
    traindata = traindata.join(problem_correct_rate, problem_correct_rate['Problem Name'] == traindata['Problem Name']).drop(
        problem_correct_rate['Problem Name'])
    testdata = testdata.join(problem_correct_rate, problem_correct_rate['Problem Name'] == testdata['Problem Name']).drop(
        problem_correct_rate['Problem Name'])
    testdata.na.fill(problem_mean_CFAR, 'Problem CFAR')

    # Step CFAR
    step_group = traindata.groupby('Step Name').count()
    step_correct_group = correct.groupby('Step Name').count()
    step_correct_rate = step_correct_group.join(step_group, step_group['Step Name'] == step_correct_group['Step Name']).drop(
        step_group['Step Name']).select('Step Name', (step_correct_group['count'] / step_group['count']).alias('Step CFAR'))
    step_mean_CFAR = step_correct_rate.select(
        mean(col('Step CFAR')).alias('mean')).collect()[0]['mean']
    traindata = traindata.join(step_correct_rate, step_correct_rate['Step Name'] == traindata['Step Name']).drop(
        step_correct_rate['Step Name'])
    testdata = testdata.join(step_correct_rate, step_correct_rate['Step Name'] == testdata['Step Name']).drop(
        step_correct_rate['Step Name'])
    testdata.na.fill(step_mean_CFAR, 'Step CFAR')

    # KC CFAR
    KC_group = traindata.groupby('KC(Default)').count()
    KC_correct_group = correct.groupby('KC(Default)').count()
    KC_correct_rate = KC_correct_group.join(KC_group, KC_group['KC(Default)'] == KC_correct_group['KC(Default)']).drop(
        KC_group['KC(Default)']).select('KC(Default)', (KC_correct_group['count'] / KC_group['count']).alias('KC CFAR'))
    KC_mean_CFAR = KC_correct_rate.select(
        mean(col('KC CFAR')).alias('mean')).collect()[0]['mean']
    traindata = traindata.join(KC_correct_rate, KC_correct_rate['KC(Default)'] == traindata['KC(Default)']).drop(
        KC_correct_rate['KC(Default)'])
    testdata = testdata.join(KC_correct_rate, KC_correct_rate['KC(Default)'] == testdata['KC(Default)']).drop(
        KC_correct_rate['KC(Default)'])
    traindata.na.fill(KC_mean_CFAR, 'KC CFAR')
    testdata.na.fill(KC_mean_CFAR, 'KC CFAR')

    # Seperate Problem Hierarchy
    traindata = traindata.withColumn(
        'Problem Unit', get_unit(traindata['Problem Hierarchy']))
    testdata = testdata.withColumn(
        'Problem Unit', get_unit(testdata['Problem Hierarchy']))

    traindata = traindata.withColumn(
        'Problem Section', get_section(traindata['Problem Hierarchy']))
    testdata = testdata.withColumn(
        'Problem Section', get_section(testdata['Problem Hierarchy']))

    traindata = traindata.withColumn(
        'KC_num', get_KCnum(traindata['KC(Default)']))
    testdata = testdata.withColumn(
        'KC_num', get_KCnum(testdata['KC(Default)']))

    traindata = traindata.withColumn(
        'Opportunity(Mean)', get_mean_opportunity(traindata['Opportunity(Default)']))
    testdata = testdata.withColumn(
        'Opportunity(Mean)', get_mean_opportunity(testdata['Opportunity(Default)']))

    traindata = traindata.withColumn(
        'Opportunity(Min)', get_min_opportunity(traindata['Opportunity(Default)']))
    testdata = testdata.withColumn(
        'Opportunity(Min)', get_min_opportunity(testdata['Opportunity(Default)']))

    naive_encoding('Anon Student Id')
    naive_encoding('Problem Name')
    naive_encoding('Problem Unit')
    naive_encoding('Problem Section')
    naive_encoding('Step Name')

    drop('Row')
    drop('Problem Hierarchy')
    drop('Step Start Time')
    drop('First Transaction Time')
    drop('Correct Transaction Time')
    drop('Step End Time')
    drop('Step Duration (sec)')
    drop('Correct Step Duration (sec)')
    drop('Error Step Duration (sec)')
    drop('Incorrects')
    drop('Hints')
    drop('Corrects')
    drop('Hints')
    drop('Opportunity(Default)')
    drop('KC(Default)')

prepare()
traindata.toPandas().to_csv('data/train_pyspark.csv', sep='\t', header=True, index = False)
testdata.toPandas().to_csv('data/test_pyspark.csv', sep='\t', header=True, index = False)
