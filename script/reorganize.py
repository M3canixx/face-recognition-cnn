import sort_arrays
def main(files_list):
    num_list = []
    for fileName in files_list:
        number = int(fileName.split('.')[0].split(' ')[-1])
        num_list.append(number)
    
    num_list, files_list = sort_arrays.main(num_list, files_list)
    return files_list