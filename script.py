import json

with open('ex.json') as f:
    json_data = json.load(f)

size = len(json_data)


def get_string(data, key):
    # i = 0
    for i in range(0, size):
        # print(data[i]['parent'])
        if data[i]['parent'] == key:
            st = ""
            # print("-->>", data[i]['requirementId'])
            st+= data[i]['requirementId'] + data[i]['initialRoom'] + data[i]['room'] + data[i]['info']['title'] + \
                   data[i]['specificationId']
            return st
    else:
        return 

        # i += 1


def get_child(data, key):
    h = 0
    liss = []
    dictt = {}
    while h < size:
        if data[h]['parent'] == key:
            # print(data[i]['requirementId'])
            # aa = data[i]['requirementId']
            # print(aa)
            # li.append(aa)
            dictt[data[h]['sortOrder']] = get_string(data, key)
            liss.append(dictt)
        h += 1
    return liss


def get_root(data):
    kk = 0
    lis = []
    substring = "_"
    while kk < size:
        if data[kk]['parent'] != None and substring in data[kk]['parent']:
            return data[kk]['requirementId']
        kk = 1
    return lis


dic = {"0": []}
root = []
dict1 = {}


def middlefun(data):
    ele = get_root(data)
    return ele


def frame_child(data):
    lis = []
    #print(data)
    # lis += get_root(data)
    # print(lis)
    ele = middlefun(data)
    i = 0
    print(ele)
    if ele == None:
        return
    else:
        i += 1
        #print(ele)
        j = 0
        while j < size:
            if data[j]['requirementId'] == ele:
                print(data[j]['requirementId'])
                nam = data[j]['sortOrder']
                #print(nam)
                # print(json_data[j]['sortOrder'])
                dict1[nam] = get_string(data, data[j]['specificationId'])
                #print(dict1)
            j += 1
            frame_child(data[j])
    #print(len(data))
    root.append(dict1)
    dic['0'] += root
    return dict1

if __name__ == "__main__":
    v=frame_child(json_data)
    #frame_child(json_data)
    #print(dic1)

