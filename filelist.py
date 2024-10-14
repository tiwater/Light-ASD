for dataType in ['trainval', 'test']:
    fileList = open('%s/%s_file_list.txt'%('/Volumes/Linux/Projects/ticos/Light-ASD/AVADataPath_bak/csv', dataType)).read().splitlines()   
    for fileName in fileList:
        print("https://s3.amazonaws.com/ava-dataset/%s/%s"%(dataType, fileName))