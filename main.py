import argparse
from solver import SOLVER

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--Train_File'     , default = 'train-v1.1.json')
    parser.add_argument('--Dev_File'       , default = 'dev-v1.1.json')
    parser.add_argument('--Prediction_File', default = 'dev_pred.json')

    parser.add_argument('--Char_Dim', default = 8  , type=int)
    parser.add_argument('--Word_Dim', default = 300, type=int)

    parser.add_argument('--Char_Channel_Width', default = 5  , type=int)
    parser.add_argument('--Char_Channel_Num'  , default = 100, type=int)
    parser.add_argument('--Max_Token_Length'  , default = 500, type=int)
    
    parser.add_argument('--Batch_Size'    , default = 20   , type=int)
    parser.add_argument('--Epoch'         , default = 12   , type=int)
    parser.add_argument('--Dropout'       , default = 0.2  , type=float)
    parser.add_argument('--EMA'           , default = 0.999, type=float)
    parser.add_argument('--GPU'           , default = 1    , type=int)
    parser.add_argument('--Learning_Rate' , default = 0.5  , type=float)
    parser.add_argument('--Exp_Decay_Rate', default = 0.999, type=float)
    

    args = parser.parse_args()

    solver = SOLVER(args)
    solver.train()
    

if __name__ == '__main__':
    main()
