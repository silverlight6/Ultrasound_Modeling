from DataFilePlayGround_2 import output2DImages
from TBIEvaluator import Cardiac_Model
from SortProcess import main as sortFile
from TBI_ResNest import main as model1
from TBISegNet_2 import main as model2

for i in range(0, 10):
    output2DImages(i)
    model1()
    sortFile()
    model2()
    Cardiac_Model()
