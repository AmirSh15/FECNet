import pandas as pd

def grade_mode(list):

    list_set = set(list)
    frequency_dict = {}
    for i in list_set:
        frequency_dict[i] = list.count(i)
    grade_mode = []
    for key, value in frequency_dict.items():
        if value == max(frequency_dict.values()):
            grade_mode.append(key)
    return grade_mode


def creat_label():
    FEC_dataset_path = 'data/FEC_dataset/faceexp-comparison-data-train-public.csv'
    dataset = pd.read_csv(FEC_dataset_path, header=None, error_bad_lines=False)

    new_dataset = {}
    names1 = []
    names2 = []
    names3 = []
    types = []
    modes = []
    for i in range(len(dataset)):
        name1 = "data/train/" + dataset.iloc[i, 0].split('/')[-1]
        name2 = "data/train/" + dataset.iloc[i, 5].split('/')[-1]
        name3 = "data/train/" + dataset.iloc[i, 10].split('/')[-1]
        the_type = dataset.iloc[i, 15]
        mode = grade_mode(
            [dataset.iloc[i, 17], dataset.iloc[i, 19], dataset.iloc[i, 21], dataset.iloc[i, 23], dataset.iloc[i, 25],
             dataset.iloc[i, 27]])
        names1.append(name1)
        names2.append(name2)
        names3.append(name3)
        types.append(the_type)
        modes.append(mode[0])


    new_dataset[0]=names1
    new_dataset[1]=names2
    new_dataset[2]=names3
    new_dataset[3]=types
    new_dataset[4]=modes
    new_data = pd.DataFrame(new_dataset)

    new_data.to_csv('data/labels.csv')
