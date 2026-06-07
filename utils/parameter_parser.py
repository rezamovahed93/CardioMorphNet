# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:12:19 2024

@author: reza
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:12:19 2024

@author: reza
"""

import argparse

def parse_args_data_splitting():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_paths", metavar='m2m_w', type=str , help="The path of obatined CSV file for the dataset")

    args = parser.parse_args()
    return args

def parse_args_preprocess():

    description = """
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    The intial implementation of the proposed framework 
    Reza Akbari Movahed
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    optional_args = parser._action_groups.pop()
    
    parser.add_argument("--data_path", metavar='dp', type=str, help="The path of the dataset in NIFTI format")    
    parser.add_argument("--preprocessed_data_path", metavar='dp', type=str, help="The path for the preprocessed data" )
    parser.add_argument('--zero_pad_flag', action='store_true', help='A boolean flag for zero padding (True if specified, False otherwise)')
    parser.add_argument("--csv_paths", metavar='m2m_w', type=str , help="The path of obatined CSV file for dataset")


    parser._action_groups.append(optional_args)

    # parser.print_help()

    args = parser.parse_args()
    return args




def parse_args_train_reg_search():

    description = """
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    The intial implementation of the proposed framework 
    Reza Akbari Movahed
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    optional_args = parser._action_groups.pop()
    
    parser.add_argument("--data_path", metavar='dp', type=str, help="The path of the preprocessed the dataset")    
    parser.add_argument("--epoch_size", metavar='epcoh', type=int, default=1, help="Epoch size (default: 50)")
    parser.add_argument("--batch_size", metavar='bs', type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--learning_rate", metavar='lr', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", metavar='pat', type=int, default=5, help="Number of epochs to wait if no improvement in validation loss is observed")
    parser.add_argument("--min_delta", metavar='delta', type=float, default=0.0001, help="Minimum change in the monitored quantity to qualify as an improvement in validation")
    parser.add_argument("--num_workers", metavar='n_workers', type=int, default=0, help="Number of workers for data loading (default: 0)")
    parser.add_argument("--model_weights_path", metavar='model_w', type=str,  default=r"./model_weights_seg_only/model_weights_seg_only.pth")
    parser.add_argument('--initial_coef_smooth_loss', type=float, default=1e-1)
    parser.add_argument('--num_multipications_for_coef_smooth_loss', type=int, default=10)


    parser._action_groups.append(optional_args)

    # parser.print_help()

    args = parser.parse_args()
    return args


def parse_args_train():

    description = """
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    The intial implementation of the proposed framework 
    Reza Akbari Movahed
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    optional_args = parser._action_groups.pop()
    
    parser.add_argument("--data_path", metavar='dp', type=str, help="The path of the preprocessed dataset")    
    parser.add_argument("--epoch_size", metavar='epcoh', type=int, default=150, help="Epoch size (default: 50)")
    parser.add_argument("--batch_size", metavar='bs', type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--learning_rate", metavar='lr', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--coeff_smoothness", metavar='cosm', type=float, default=0.03, help="Smoothness coefficient (default: 0.03)")
    parser.add_argument('--seg_freez', action='store_true', help='Freezing the SegNet model')
    parser.add_argument("--patience", metavar='pat', type=int, default=5, help="Number of epochs to wait if no improvement in validation loss is observed")
    parser.add_argument("--min_delta", metavar='delta', type=float, default=0.0001, help="Minimum change in the monitored quantity to qualify as an improvement in validation")
    parser.add_argument("--seg_model_path", metavar='model_w', type=str,  default=r"./seg_model/")
    parser.add_argument("--num_workers", metavar='n_workers', type=int, default=0, help="Number of workers for data loading (default: 0)")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument("--model_weights_path", metavar='model_w', type=str,  default=r"C:\My PHD Codes\My Implementation\First Proposed Method\Final Implementation (1)\best_model_epoch(43) val(0.31).pth")

    parser._action_groups.append(optional_args)

    # parser.print_help()

    args = parser.parse_args()
    return args

def parse_args_test():

    description = """
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    The intial implementation of the proposed framework 
    Reza Akbari Movahed
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    optional_args = parser._action_groups.pop()
    
    parser.add_argument("--data_path", metavar='dp', type=str, help="The path of the preprocessed the dataset")    
    parser.add_argument("--batch_size", metavar='bs', type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--model_weights_path", metavar='model_w', type=str,  default=r"C:\My PHD Codes\My Implementation\First Proposed Method\Final Implementation (1)\best_model_epoch(43) val(0.31).pth")
    parser.add_argument("--saving_results_path", metavar='results_path', type=str,  default=r"C:\My PHD Codes\My Implementation\First Proposed Method\Final Implementation (1)\results.pkl")
    parser.add_argument("--seg_model_path", metavar='model_w', type=str,  default=r"./seg_model/")

    parser._action_groups.append(optional_args)

    # parser.print_help()

    args = parser.parse_args()
    return args


