from operator import index

list = [
    'ade',
    'kenny',
    'gia',
    'monica',
    'neo'
]

def print_list():
    for item in list:
        print(item)

new_list = ['monaco', 'bahrain', 'singapore', 'deutchsland']
for item in new_list:
    list.append(new_list[new_list.index(item)])
print_list()
