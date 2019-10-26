import copy
import random

def findIndex(array1, key, value):
    return find_index(array1, key, value)

def find_index(array1, key, value):
    for index, arr_item in enumerate(array1):
        if key in arr_item and arr_item[key] == value:
            return index
    return -1

def extend_object(default, new):
    final = {}
    # Go through defaults first
    for key in default:
        if key not in new:
            final[key] = default[key]
        else:
            final[key] = new[key]
    # In case any keys in new but not in default, add them
    for key in new:
        if key not in final:
            final[key] = new[key]
    return final

def sort2D(array1, key, order = 'ascending'):
    if len(array1) < 2:
        return array1

    # def compare(a, b):
    #     aVal = a[key]
    #     bVal = b[key]
    #     if aVal == bVal:
    #         return 0
    #     if (aVal > bVal and order == 'ascending') or (aVal < bVal and order == 'descending'):
    #         return 1
    #     return -1
    def getValue(item):
        return item[key]

    reverse = True if order == 'descending' else False
    return sorted(array1, key=getValue, reverse=reverse)

def omit(object1, keys = []):
    new_object = {}
    for key in object1:
        if key not in keys:
            new_object[key] = object1[key]
    return new_object

def pick(object1, keys = []):
    new_object = {}
    for key in object1:
        if key in keys:
            new_object[key] = object1[key]
    return new_object

def map_pick(array1, keys = []):
    def pick1(obj1):
        return pick(obj1, keys)

    return list(map(pick1, array1))

def mapOmit(array1, omitKeys = []):
    def omit1(obj1):
        return omit(obj1, omitKeys)

    return list(map(omit1, array1))

def get_key_array(items, key, skipEmpty=0, emptyValue=None):
    if skipEmpty:
        return list(map(lambda item: item[key] if key in item else emptyValue, items))
    else:
        return list(map(lambda item: item[key], items))

# def append_if_unique(array1, value):
#     if value not in array1:
#         array1.append(value)

def random_string(length = 10):
    text = ''
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    chars_length = len(chars)
    counter = 0
    while counter < length:
        index = random.randint(0, (chars_length - 1))
        text = text + chars[index]
        counter = counter + 1
    return text

def removeArrayIndices(array, indices):
    array1 = copy.deepcopy(array)
    for index, item in reversed(list(enumerate(array1))):
        if index in indices:
            del array1[index]
    return array1
