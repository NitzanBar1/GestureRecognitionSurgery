#!/usr/bin/python2.7
import os
import argparse
import torch
from model import Trainer
import os
from clearml import Task, Logger
import random
import numpy as np
from batch_gen import BatchGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 44


def reweighting_loss_calculation(labels_path, labels, technique='IPW'):
    files_labels = [f for f in os.listdir(labels_path) if os.path.isfile(os.path.join(labels_path, f))]
    class_hist = {k: 0 for k in labels}
    for label_file in files_labels:
        with open(os.path.join(labels_path, label_file), 'r') as file:
            content = [(line.split()[-1], int(line.split()[1]) - int(line.split()[0]) - 1) for line in file.readlines()]
            for class_label, amount in content:
                class_hist[class_label] += amount
    if technique == 'ISNS':
        class_weights = {k: 1 / (v) ** 0.5 for k, v in
                      class_hist.items()}  # ISNS -Inverse of Square Root of Number of Samples
    else:
        class_weights = {k: v / sum([x[1] for x in class_hist.items()]) * len(class_hist.items()) for k, v in
                      class_hist.items()}  # IPW -Inverse of probability of Number of Samples
    weigths = torch.tensor(
        [weight for _, weight in sorted([(k, v) for k, v in class_weights.items()], key=lambda x: x[0])])
    return weigths.float().to(DEVICE)


def fold_split(features_path, val_path, test_path):
    with open(val_path, 'r') as f:
        val_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]
    with open(test_path, 'r') as f:
        test_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]
    train_files = [f for f in os.listdir(features_path) if os.path.isfile(os.path.join(features_path, f))]
    train_files = list(set(train_files) - set(test_files + val_files))
    return train_files, val_files, test_files


def initialize_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


def run_train(val_path_fold, test_path_fold, features_path_fold, kinematic_features_path, results_dir, model_dir, actions_dict,
              num_layers_PG, num_layers_R, num_R, num_f_maps,
              num_epochs, batch_size, lr, features_dim, clogger, sample_rate, true_labels_dir,
              model_type='mstcn2', sample_size=5,
              transformer_params = {},
              concat_kinematic_data=False,
			  label_class_weights=None, weighted_flag=False,
			  ridge_reg=False, attention=False,
              lstm_att=False):
    fold_num = features_path_fold.split("/")[-2]
    print(f"\t{fold_num}")
    labels = ['G0', 'G1', 'G2', 'G3', 'G4', 'G5']
    if weighted_flag:
        class_weights = reweighting_loss_calculation(true_labels_dir, labels=labels, technique='IPW')
    else:
        class_weights = None

    num_classes = len(labels)

    # Experiment flags
    folder_class_labels_flag = "ClassWeighted" if label_class_weights else "NotClassWeighted"
    sample_size_flag = f"Sample size {sample_size}"
    LSTM_Att_flag = "LSTM_Att" if lstm_att else "Without_LSTM_Att"
    exts = [sample_size_flag, LSTM_Att_flag, folder_class_labels_flag]
    ########################################################
    folder_name = results_dir.format(fold_num, "_".join(exts))
    model_folder_name = model_dir.format(fold_num, "_".join(exts))
    try:
        os.makedirs(folder_name)
        os.makedirs(model_folder_name)
    except:
        print("Could not create the experiment folders")

    # Split data into folds
    vid_list_file, vid_list_file_val, vid_list_file_test = fold_split(features_path_fold, val_path_fold,
                                                                      test_path_fold)
    # Generate batches for train
    batch_gen_train = BatchGenerator(num_classes, actions_dict, true_labels_dir, features_path_fold, kinematic_features_path, sample_rate)
    batch_gen_train.read_data(vid_list_file)

    # Generate batches for validation
    batch_gen_val = BatchGenerator(num_classes, actions_dict, true_labels_dir, features_path_fold, kinematic_features_path, sample_rate)
    batch_gen_val.read_data(vid_list_file_val)

    # create trainer instance
    trainer = Trainer(num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R, num_f_maps=num_f_maps, dim=features_dim, num_classes=num_classes,
                      split=fold_num,
                      model_type=model_type,
                      sample_size=sample_size, transformer_params=transformer_params,
                      concat_kinematic_data=concat_kinematic_data, kinematic_features_path=kinematic_features_path,
					  weighted=weighted_flag, class_weights=class_weights if label_class_weights else None,
					  ridge_reg=ridge_reg, attention=attention, lstm_att=lstm_att)
    # train the model
    trainer.train(model_folder_name, batch_gen_train, batch_gen_val, num_epochs=num_epochs, batch_size=batch_size,
                  learning_rate=lr, device=DEVICE, clogger=clogger, lambda_l2=0.001)

    trainer.predict(model_folder_name, folder_name,
                    features_path_fold, vid_list_file_test, num_epochs, actions_dict, DEVICE,
                    sample_rate)


def add_all_arguments(parser):
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_layers_PG', default=10, type=int)
    parser.add_argument('--num_layers_R', default=10, type=int)
    parser.add_argument('--num_R', default=3, type=int)
    parser.add_argument('--true_labels_dir', default='/datashare/APAS/transcriptions_gestures/', type=str)
    parser.add_argument('--action', default='train')
    parser.add_argument('--split', default='1')
    parser.add_argument('--features_dim', default='1280', type=int)
    parser.add_argument('--batch_size', default='1', type=int)
    parser.add_argument('--lr', default='0.0005', type=float)
    parser.add_argument('--num_f_maps', default='65', type=int)
    parser.add_argument('--mapping_file', default='/datashare/APAS/mapping_gestures.txt', type=str)
    parser.add_argument('--sample_rate', default=1, type=int)


def get_args(args):
    return args.num_epochs, args.features_dim, args.batch_size, args.lr, args.true_labels_dir, args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, args.mapping_file, args.sample_rate


def main():
    initialize_seed()
    parser = argparse.ArgumentParser()
    add_all_arguments(parser)
    args = parser.parse_args()
    num_epochs, features_dim, batch_size, lr, true_labels_dir, num_layers_PG, num_layers_R, \
    num_R, num_f_maps, mapping_file, sample_rate = get_args(args)
    folds_split_directories = [(f"/datashare/APAS/folds/valid {i}.txt",
                                f"/datashare/APAS/folds/test {i}.txt",
                                f"/datashare/APAS/features/fold{i}/") for i in range(5)]
    kinematic_features_path = "/datashare/APAS/kinematics_npy"

    transformer_params = dict(num_decoders=3, num_layers=9, r1=2, r2=2, num_f_maps=num_f_maps, input_dim=features_dim, num_classes=6,
                channel_masking_rate=0.25, lstm_dropout=0.35)

    try:
        latest = max([int(exp.split("exp")[-1]) for exp in os.listdir("./results")])
    except Exception:
        latest = 0

    new_exp = f"exp{latest + 1}"
    print(f"Experiment: {new_exp}")
    results_dir = "./results/" + str(new_exp) + "/{}/{}"
    model_dir = "./models/" + str(new_exp) + "/{}/{}"

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}

    if args.action == "baseline":
        task = Task.init(project_name='CVSA - Final project', task_name=new_exp + " Baseline")
    elif args.action == "train":
        task = Task.init(project_name='CVSA - Final project', task_name=new_exp + " Chosen model")
    else:
        task = Task.init(project_name='CVSA - Final project', task_name=new_exp)

    clogger = task.get_logger()

    print("Starting training!")
    if args.action == "train_tradeoff":
        for val_path_fold, test_path_fold, features_path_fold in folds_split_directories:
            for sample_size in [1, 5, 10, 30, 60]:
                run_train(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                          features_path_fold=features_path_fold, kinematic_features_path=kinematic_features_path,
						  results_dir=results_dir,
                          model_dir=model_dir,
                          actions_dict=actions_dict,
                          num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                          num_f_maps=num_f_maps, num_epochs=num_epochs, true_labels_dir=true_labels_dir,
                          batch_size=batch_size, lr=lr, features_dim=features_dim,
                          clogger=clogger, sample_rate=sample_rate,
						  label_class_weights=True, weighted_flag=False,
                          ridge_reg=True,
						  model_type='mstcn2', sample_size=sample_size)
						  
    if args.action == "train":
        for val_path_fold, test_path_fold, features_path_fold in folds_split_directories:
            run_train(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                      features_path_fold=features_path_fold, kinematic_features_path=kinematic_features_path,
                      results_dir=results_dir,
                      model_dir=model_dir,
                      actions_dict=actions_dict,
                      num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                      num_f_maps=num_f_maps, num_epochs=num_epochs, true_labels_dir=true_labels_dir,
                      batch_size=batch_size, lr=lr, features_dim=features_dim,
                      clogger=clogger, sample_rate=sample_rate,
					  model_type='mstcn2', sample_size=1,                      
                      label_class_weights=False, weighted_flag=False,
                      ridge_reg=False, attention=True,
                      lstm_att=True)

    if args.action == "baseline":
        for val_path_fold, test_path_fold, features_path_fold in folds_split_directories:
            run_train(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                      features_path_fold=features_path_fold, kinematic_features_path=kinematic_features_path,
                      results_dir=results_dir,
                      model_dir=model_dir,
                      actions_dict=actions_dict,
                      num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                      num_f_maps=num_f_maps, num_epochs=num_epochs, true_labels_dir=true_labels_dir,
                      batch_size=batch_size, lr=lr, features_dim=features_dim,
                      clogger=clogger, sample_rate=sample_rate,
                      model_type='mstcn2', sample_size=1,
                      label_class_weights=False, weighted_flag=False,
                      ridge_reg=False, attention=False,
                      lstm_att=False)

    if args.action == "mstcn2_concat_kinematic":
        for val_path_fold, test_path_fold, features_path_fold in folds_split_directories:
            run_train(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                        features_path_fold=features_path_fold, kinematic_features_path=kinematic_features_path, 
                        results_dir=results_dir,
                        model_dir=model_dir,
                        actions_dict=actions_dict,
                        num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                        num_f_maps=num_f_maps, num_epochs=num_epochs, true_labels_dir=true_labels_dir,
                        batch_size=batch_size, lr=lr, features_dim=features_dim,
                        clogger=clogger, sample_rate=sample_rate,
						label_class_weights=None, weighted_flag=False,
					  	ridge_reg=False,
                        model_type='mstcn2', sample_size=1,
                        concat_kinematic_data=True)

    if args.action == "transformer":
         for val_path_fold, test_path_fold, features_path_fold in folds_split_directories:
            run_train(val_path_fold=val_path_fold, test_path_fold=test_path_fold,
                      features_path_fold=features_path_fold, kinematic_features_path=kinematic_features_path, 
                      results_dir=results_dir,
                      model_dir=model_dir,
                      actions_dict=actions_dict,
                      num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R,
                      num_f_maps=num_f_maps, num_epochs=num_epochs, true_labels_dir=true_labels_dir,
                      batch_size=batch_size, lr=lr, features_dim=features_dim,
                      clogger=clogger, sample_rate=sample_rate,
					  label_class_weights=None, weighted_flag=False,
					  ridge_reg=False,
                      model_type='asformer', sample_size=1,
                      transformer_params=transformer_params,
                      concat_kinematic_data=False)


if __name__ == '__main__':
    print("start running")
    main()