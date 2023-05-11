from dataloading_train import load_training_data
from dataloading_test import load_testing_data
from segmentation_train import train_loop_seg
from segmentation_test import test_model, calculateDice

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--patch_size', nargs='+', type=int, default=[64,64,64],help="Size of 3D boxes cropped out of CT volumes as model input")
    parser.add_argument('--training_data_path', type=str, help="Set the path to training dataset")
    parser.add_argument('--testing_data_path', type=str, help="Set the path to testing dataset")
    parser.add_argument('--testing_data_solution_path', type=str,
                        help="Set the path to solution of testing dataset")
    parser.add_argument('--lr', default=0.01, type=float, help="Learning rate")
    parser.add_argument('--train', type=bool, default=False, help="Use True for training")
    parser.add_argument('--test', type=bool, default=False, help="Use True for testing")
    parser.add_argument('--model_path', type=str, help="Set the path of the model to be tested")
    parser.add_argument('--num_classes', type=int, default=1, help="Number of classes / number of output layers")
    args = parser.parse_args()
    print(args)

    # The following code is an example of data preprocessing and training algorithm and can be used as starting point.
    if not args.train and not args.test:
        raise TypeError(
            "Please specify, whether you want to run the training or testing code by setting the parameter --train=True or --test=True")
    if args.train:
        if not args.training_data_path == None:
            # Preprocess training data. When first time called, data is preprocessed and saved to "my_training_data".
            # When this folder exists, data is loaded from it directly.
            train_loader, val_loader = load_training_data(args)
            print("Number of samples in datasets:")
            print(" training: " + str(len(train_loader.dataset)))
            print(" validation: " + str(len(val_loader.dataset)))
            print("Shape of data:")
            print(" image: " + str(next(iter(train_loader))[0].shape))
            print(" ROI mask: " + str(next(iter(train_loader))[1].shape))
            # Train model and save best performing model at model_path.
            model_path = train_loop_seg(train_loader, val_loader, args)
            if args.test and not args.testing_data_path == None:
                # Preprocess testing data. When first time called, data is preprocessed and saved to "my_testing_data".
                # When this folder exists, data is loaded from it directly.
                test_loader = load_testing_data(args)
                print("Number of samples in datasets:")
                print(" testing: " + str(len(test_loader.dataset)))
                args.model_path = model_path
                # Testing data is being predicted and predictions are being saved in folder "testing_data_prediction_segmentation".
                test_model(test_loader, args)
                if not args.testing_data_solution_path == None:
                    # Accuracy metric is being calculated between data in folder args.testing_data_solution_path and "testing_data_prediction_segmentation".
                    dc = calculateDice(args)
                    print("Dice on testing dataset with entire CT volumes: " + str(dc))
        else:
            raise TypeError(
                "Please specify the path to the training data by setting the parameter --training_data_path=\"path_to_trainingdata\"")
    elif args.test:
        if args.model_path == None:
            raise TypeError("Please specify the path to model by setting the parameter --model_path=\"path_to_model\"")
        else:
            if not args.testing_data_path == None:
                # Preprocess testing data. When first time called, data is preprocessed and saved to "my_testing_data".
                # When this folder exists, data is loaded from it directly.
                test_loader = load_testing_data(args)
                # Testing data is being predicted and predictions are being saved in folder "testing_data_prediction_segmentation".
                test_model(test_loader, args)
                if not args.testing_data_solution_path == None:
                    # Accuracy metric is being calculated between data in folder args.testing_data_solution_path and "testing_data_prediction_segmentation".
                    dc = calculateDice(args)
                    print("Dice on testing dataset with entire CT volumes: " + str(dc))
                else:
                    raise TypeError(
                        "Please specify the path to the testing solution/ground truth data by setting the parameter --testing_data_solution_path=\"path_to_testingdata_solution\"")
            else:
                raise TypeError(
                    "Please specify the path to the testing data by setting the parameter --testing_data_path=\"path_to_testingdata\"")
