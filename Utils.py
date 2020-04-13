import numpy as np
from config import window_size,results_path
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import fitz

class Generate_Dataset():
    def __init__(self,Dataset,window_size):
        self.Dataset = Dataset
        self.window_size = window_size
    def chopper(self):
      chop=list(range(self.window_size))
      chDataset=np.delete(self.Dataset,chop,0)
      return chDataset
      
    def sequencer(self):
      sample=0
      sDataset=[]
      SDataset=[]
      while sample<len(self.Dataset)-self.window_size:
        for time_step in range(self.window_size):
            sDataset.append(self.Dataset[sample+time_step])
        sample+=1
        SDataset.append(sDataset)
        sDataset=[]
      return SDataset

    def LSTM_dataset(self):
        sequenced=self.sequencer()
        DatasetLSTM=np.asarray(sequenced)
        return DatasetLSTM
    def CNN_dataset(self):
        DatasetCNN = self.chopper()
        return DatasetCNN

    def generate(self):
        Ts_DatasetCNN = self.CNN_dataset()
        Ts_DatasetLSTM = self.LSTM_dataset()
        X_testCNN=np.reshape(Ts_DatasetCNN,(Ts_DatasetCNN.shape[0],Ts_DatasetCNN.shape[1],Ts_DatasetCNN.shape[2],1))
        X_testLSTM=np.reshape(Ts_DatasetLSTM,(Ts_DatasetLSTM.shape[0],Ts_DatasetLSTM.shape[1],Ts_DatasetLSTM.shape[2],Ts_DatasetLSTM.shape[3],1))
        return X_testCNN,X_testLSTM
class SplitPDF():
    def __init__(self,PDF_Path,EOD):
        self.PDF_Path = PDF_Path
        self.EOD = EOD
        self.EODIndx = []
        self.EOD2Indx()
    def EOD2Indx(self):
        indx = window_size
        for EOD in self.EOD:
            if EOD ==1:
                self.EODIndx.append(indx)
            indx+=1
    def split(self):
        doc = fitz.open(self.PDF_Path)
        for i in range(1,len(self.EODIndx)):
            if i == 1:
                newDoc = fitz.open()
                newDoc.insertPDF(doc, to_page = self.EODIndx[0])
                newDoc.save(results_path+str((self.EODIndx[0]+1))+'.pdf')
                newDoc.close()
            newDoc = fitz.open()
            newDoc.insertPDF(doc, from_page = self.EODIndx[i-1]+1 ,to_page = self.EODIndx[i])
            newDoc.save(results_path+str((self.EODIndx[i]+1))+'.pdf')
            newDoc.close()
