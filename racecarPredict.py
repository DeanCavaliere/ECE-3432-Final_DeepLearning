from av_nn_tools import NNTools
import time

start_time = time.time()

TRAIN_DATA = './data/list/dummy.csv'
TEST_DATA = './data/list/test_1.csv'
TrainTimeFile = 'TrainTimeData_Amd1700CPU.csv'

SERVO_TRAIN_SETTING = "data/set_servo_train.json"
SERVO_TEST_SETTING = "data/set_servo_test.json"
SERVO_MODEL = 'models/servo_model.pth'
TestTimeFile = 'TestTimeData_Amd1700CPU.csv'
#MOTOR_TRAIN_SETTING = "data/set_motor_train.json"
#MOTOR_TEST_SETTING = "data/set_motor_test.json"
#MOTOR_MODEL = 'models/motor_model.pth'

IMAGE_FILE = "data/images/03_12_2020_0/output_0002/i0001053_s17_m17.jpg"

servo_test = NNTools(SERVO_TEST_SETTING)
servo_test.load_model(SERVO_MODEL)
servo_test.test(TEST_DATA, TestTimeFile)
pred_time = time.time()
servov = servo_test.predict(IMAGE_FILE)
print("--- Total Prediction Time is %s seconds ---" % (time.time() - pred_time))
print(servov)
print('\n')
print("--- Total Runtime is %s seconds ---" % (time.time() - start_time))