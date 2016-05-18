
'''
Loads the file posnegdata, which is a three-column csv in the form


	entry   |    positiv    |    negativ    


where entry is the word, positiv has the value "Positiv" if the word
has a positive sentiment,"Negativ" if the word has a negative sentiment,
and NULL otherwise.

'''


import csv
def loadPosNegList(filename="train.csv"):

	posList = []
	negList = []

	with open(filename, 'r') as inputFile:
		next(inputFile)
		reader = csv.reader(inputFile, delimiter=',')
		index = 0
		for entry, positiv, negativ in reader:
			if "#" in entry:
				entry = entry.split("#")[0]
			if positiv:
				posList.append(entry)
			elif negativ:
				negList.append(entry)

	return (posList, negList)