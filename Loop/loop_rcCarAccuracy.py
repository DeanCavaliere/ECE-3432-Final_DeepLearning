import json
import pandas as pd

from av_nn_tools import NNTools
from av_parse_data import ParseData

class accuracyTest:
    def __init__(self, settings='./data/set_accuracy_test.json'):

        self.TEST_LIST = "./data/list/final_test.csv"
        self.SETTINGS = "./data/set_accuracy_test.json"

    def beginAccTest(self):
        data = pd.read_csv(self.TEST_LIST)
        parsedata = ParseData()
        with open(self.SETTINGS) as fp:
            content = json.load(fp)

            shape = content['shape']
            servo_pred = NNTools(content["servo_setting"])
            servo_pred.load_model(content['servo_model'])

        servo_count = 0

        for index in range(len(data)):
            _, servo, motor = parsedata.parse_data(data["image"][index])

            pred_servo = servo_pred.predict(data["image"][index])

            if abs(servo - pred_servo) <= content['error_margin']:
                # print(servo)
                servo_count += 1
        totalAcc = (100 * servo_count / (index + 1))
        return totalAcc