import json

with open('data/set_servo_train.json') as f:
       data = json.load(f)

print(data)
print(data['optimizer'])