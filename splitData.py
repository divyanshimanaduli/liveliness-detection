import os
import random
import shutil  #offers high-level file operations such as copying and deleting files.
from itertools import islice #iterating over a subset of elements

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake","real"]

try:
    shutil.rmtree(outputFolderPath) #This attempts to delete the entire directory tree at the path specified by
    # outputFolderPath. If the directory does not exist, this will raise an OSError
except OSError as e:
    os.mkdir(outputFolderPath) #If an OSError is caught (indicating the directory did not exist and could not
    # be removed), this creates the directory specified by outputFolderPath.

# --------  Directories to Create -----------
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# --------  Get the Names  -----------
listNames = os.listdir(inputFolderPath) #retrieves a list of filenames from the specified directory.

# --------extracting unique image names (without extensions) from the initial list of filenames ---------#
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))

# --------  Shuffle -----------
random.shuffle(uniqueNames)

# --------  Find the number of images for each folder -----------
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# --------  Put remaining images in Training -----------
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining

# --------  Split the list -----------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]#takes the Input sequence and extracts a specific
# portion of it based on a length provided by another list (lengthToSplit). It then converts that extracted portion
# into a regular list.

print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

# --------  Copy the files  -----------

sequence = ['train', 'val', 'test']
for i,out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        #copies the corresponding image file (.jpg) from the inputFolderPath to the images subdirectory within the
        # current split's directory (train, val, test) in the outputFolderPath

        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process Completed...")


# -------- Creating Data.yaml file  -----------

dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")