import json

with open('ex.json') as f:
    json_data = json.load(f)

size = len(json_data)
# print(size)
j = 0


def get_string(data, key):
    # i = 0
    for i in range(0, size):
        # print(data[i]['parent'])
        if data[i]['parent'] == key:
            str = ""
            # print("-->>", data[i]['requirementId'])
            str += data[i]['requirementId'] + data[i]['initialRoom'] + data[i]['room'] + data[i]['info']['title'] + \
                   data[i]['specificationId']
            return str

        # i += 1


li = []


def get_child(data, key):
    h = 0
    liss = []
    dictt = {}
    while h < size:
        if data[h]['parent'] == key:
            # print(data[i]['requirementId'])
            aa = data[i]['requirementId']
            #print(aa)
            li.append(aa)
            dictt[data[h]['sortOrder']] = get_string(data, key)
            liss.append(dictt)
        h += 1
    return liss



# s = {}
# s = set()

dict = {"0": []}

while j < size:
    key = json_data[j]['specificationId']
    # print(json_data[j]['requirementId'])
    # print(get_string(json_data, key))
    # print("--------------------------")
    j += 1

# for Root Values
k = size - 1
lis = []
kk = 0
substring = "_"
while kk < size:
    if json_data[kk]['parent'] != None and substring in json_data[kk]['parent']:
        # print(json_data[kk]['requirementId'])
        lis.append(json_data[kk]['requirementId'])
    kk += 1

lisSize = len(lis)
dic = {"0": []}
root = []
dict1 = {}

for ele in lis:
    j = 0

    while j < size:
        if json_data[j]['requirementId'] == ele:
            nam = json_data[j]['sortOrder']
            # print(json_data[j]['sortOrder'])
            dict1[nam] = get_string(json_data, json_data[j]['specificationId'])
        j += 1

# print(dict1)
root.append(dict1)
dic['0'] += root
dict["0"] += root
# print(dic)

i = 0
a = 0
for i in range(0, lisSize):  # is size of list which contains root ele
    for j in range(0, len(json_data)):  # containing the json size
        # print(lis[i])
        if json_data[j]['requirementId'] == lis[i]:
            key = json_data[j]['specificationId']
            name = json_data[j]['sortOrder']
            dict[name] = []
            # get_child(json_data,key)
            dict[name] += get_child(json_data, key)

print(dict)

