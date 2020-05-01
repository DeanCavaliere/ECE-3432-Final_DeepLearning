from av_nn_tools import NNTools
import time
import pandas as pd
from rcCarAccuracy import accuracyTest
from datetime import datetime

testRuns = 10;

TRAIN_DATA = './data/list/dummy.csv'
TEST_DATA = './data/list/test_1.csv'
TrainTimeFile = 'TrainTimeData_Amd1700CPU.csv'

SERVO_TRAIN_SETTING = "data/set_servo_train.json"
SERVO_TEST_SETTING = "data/set_servo_test.json"
SERVO_MODEL = 'models/servo_model.pth'
pathEnd = '.pth'
TestTimeFile = 'TestTimeData_Amd1700CPU.csv'
SERVO_DUMMY = 'models/servo_model_'
#MOTOR_TRAIN_SETTING = "data/set_motor_train.json"
#MOTOR_TEST_SETTING = "data/set_motor_test.json"
#MOTOR_MODEL = 'models/motor_model.pth'

IMAGE_FILE = "data/images/03_12_2020_0/output_0002/i0001053_s17_m17.jpg"

ResultsCSV = open('ResultsCSV', 'w')

df = pd.DataFrame({'Test Run #': [''], 'Accumulated Loss': [''], 'Accuracy': ['']})
df.to_csv(ResultsCSV, mode='a', header=True, index=False)

for i in range(testRuns):

    dateTimeObj = datetime.now()
    timeObj = dateTimeObj.time()
    print("<<<<-----         Test run # " + str(i + 1) + ' started at ' + str(timeObj.hour) + ':' + str(timeObj.minute) +'          ----->>>>')

    SERVO_MODEL_NEW = SERVO_DUMMY+str(i+1)+pathEnd
    print('<<<<-----    Now Generating: ' + str(SERVO_MODEL_NEW)+'    ----->>>>')

    servo_train = NNTools(SERVO_TRAIN_SETTING)
    servo_train.load_model(SERVO_MODEL)
    servo_train.train(TRAIN_DATA, TrainTimeFile)
    servo_train.save_model(SERVO_MODEL_NEW)

    SERVO_MODEL = SERVO_MODEL_NEW

    servo_test = NNTools(SERVO_TEST_SETTING)
    servo_test.load_model(SERVO_MODEL_NEW)
    lossAccum = servo_test.test(TEST_DATA, TestTimeFile)

    servov = servo_test.predict(IMAGE_FILE)
    rcA = accuracyTest()
    totalAcc = rcA.beginAccTest()

    print(str(lossAccum) + ' Accumulated Loss for test number ' + str(i+1))
    print(str(servov) + ' for test number ' + str(i+1))
    print(str(totalAcc) + ' for test number ' + str(i+1))
    print('\n')

    df = pd.DataFrame({'Test Run #': [str(i+1)], 'Accumulated Loss': [lossAccum], 'Accuracy': [totalAcc]})
    df.to_csv(ResultsCSV, mode='a', header=False, index=False)
