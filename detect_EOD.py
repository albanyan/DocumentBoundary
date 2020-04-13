
from Utils import Generate_Dataset,SplitPDF
from config import window_size,model_path
from model import build_model,detect
from Preprocessor import Preprocessor
import argparse


parser = argparse.ArgumentParser(description='Enter A file to detect its EOD')
parser.add_argument('PDF_Path')
args = parser.parse_args()
PDF_Path = args.PDF_Path

if __name__ == '__main__':

    print("Loading and preprocessing the PDF file")
    processor = Preprocessor(PDF_Path)
    preprocessed = processor.Preprocess()

    print("File successfully loaded and processed")
    Generator = Generate_Dataset(preprocessed,window_size)
    X_testCNN,X_testLSTM = Generator.generate()

    print("Building the model")
    MainModel = build_model()
    EOD = detect(MainModel,model_path,X_testLSTM,X_testCNN)

    print("Saving the splitted files")
    splitter = SplitPDF(PDF_Path,EOD)
    splitter.split()
