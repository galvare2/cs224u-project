import json

TRAIN_OP_DATA = "cmv/op_task/train_op_data.jsonlist"

def load_data():
	f = open(TRAIN_OP_DATA, "r")
	data = []
	for line in f:
		data.append(json.loads(line))
	print data[1]['delta_label']


if __name__ == "__main__":
	load_data()