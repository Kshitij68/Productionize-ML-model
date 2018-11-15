def drop_values(data, keys):
    """
    Drop the keys which are not useful in prediction
    """
    for dictionary in data:
        for key in keys:
            if key not in dictionary.keys():
                raise Exception('The dictionary {} does not have key {}'.format(dictionary, key))
            del dictionary[key]

def get_age_bins(age):
    if age <= 20:
        return 1
    elif 20 < age <= 30:
        return 2
    elif 30 < age <= 45:
        return 3
    else:
        return 4

def get_fare_bins(fare):
    if fare <= 15:
        return 1
    elif 15 < fare <= 40:
        return 2
    elif 40 < fare <= 100:
        return 3
    else:
        return 4
