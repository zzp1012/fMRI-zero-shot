import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np

# import internal libs
from data import fMRIDataUtils
from model import ModelUtils
from utils import get_datetime, set_logger, get_logger, log_settings


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--person_id", default=1, type=int,
                        help='the person id.')
    parser.add_argument("--leave", default=["cat"], type=str, nargs="+",
                        help='the leave out class.')
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"person{args.person_id}",
                         "leave" + "-".join(args.leave)])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, verbose=args.verbose)
    # show the args.
    log_settings(args)

    # load the data
    inputs, features = fMRIDataUtils.read(person_id=args.person_id)
    # split the data
    train_inputs, train_features, test_inputs, test_features = \
        fMRIDataUtils.split(inputs, features, leave_out=args.leave)
    
    ## train the regressor part
    logger.info("#######train the regressor part.")
    # data to regression format
    train_X, train_y = fMRIDataUtils.to_regression(train_inputs, train_features)
    logger.info(f"train_X.shape: {train_X.shape}, train_y.shape: {train_y.shape}")

    # get the regressor and train
    regressor = ModelUtils.auto("lr")
    regressor = ModelUtils.train(regressor, train_X, train_y)

    # get the score
    train_score = ModelUtils.score(regressor, train_X, train_y)
    logger.info(f"regressor's train score: {train_score}")

    # predict the test data
    logger.info("#######predict the test features.")
    pred_features = {}
    for word, test_X in test_inputs.items():
        # predict the features
        test_X = np.concatenate(test_X, axis=0)
        pred_features[word] = regressor.predict(test_X)  
        logger.info(f"word: {word}, test_X.shape: {test_X.shape}, test_y.shape: {pred_features[word].shape}")

    ## train the classifier part
    logger.info("#######train the classifier part.")
    # data to classification format
    if len(args.leave) == 1:
        cls_X, cls_y = fMRIDataUtils.to_classification(features)
    elif len(args.leave) == 2:
        cls_X, cls_y = fMRIDataUtils.to_classification(test_features)
    else:
        raise ValueError("leave out class should be 1 or 2.")
    logger.info(f"cls_X.shape: {cls_X.shape}, cls_y.shape: {cls_y.shape}")

    # get the classifier and train
    classifier = ModelUtils.auto("knn", n_neighbors=1)
    classifier = ModelUtils.train(classifier, cls_X, cls_y)

    ## test the classifier part
    logger.info("#######test the classifier part.")
    for word, pred_y in pred_features.items():
        # predict the label
        test_acc = ModelUtils.score(classifier, pred_y, [word] * pred_y.shape[0])
        logger.info(f"word: {word}; test_acc: {test_acc}")
    

if __name__ == "__main__":
    main()