#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
from eval import f_score, edit_score
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models, transforms as T
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MS_TCN2_Modified(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, sample_size=5, attention=False,
                 lstm=False, lstm_att=False, upsample=True):
        super(MS_TCN2_Modified, self).__init__()
        self.attention = attention
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(
                Refinement(num_layers=num_layers_R, num_f_maps=num_f_maps, num_classes=num_classes, dim=num_classes,
                           attention=attention, lstm=lstm, lstm_att=lstm_att)) for s in
                range(num_R)])
        #   Added by us
        self.gru = nn.GRU(input_size=6, batch_first=True, hidden_size=64, num_layers=3, dropout=0.1, bidirectional=True)
        self.hidden_to_label = nn.Linear(in_features=128, out_features=6)
        self.sample_size = sample_size

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for i, R in enumerate(self.Rs):
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        #####################################################
        # To do - delete or change the architecture
        # self.sample_size = 5
        # out = F.interpolate(F.softmax(out, dim=1), int(out.shape[-1] / self.sample_size))  # down sampling
        # out = self.gru(out.transpose(2, 1))[0].squeeze(0)
        # out = self.hidden_to_label(out).unsqueeze(0).transpose(1, 2)
        # outputs = torch.cat((outputs, F.interpolate(out, outputs.shape[-1]).unsqueeze(0)), dim=0)  # up sampling
        return outputs


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, weighted=False):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])
        #   Added by us
        self.weighted = weighted
        self.Ws = nn.ParameterList(
            [copy.deepcopy(nn.Parameter(data=torch.rand(1), requires_grad=True)) for s in range(num_R + 1)])
        self.hidden_to_label = nn.Linear(in_features=128, out_features=6)


    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for i, R in enumerate(self.Rs):
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()
        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i), dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        ))
        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i)
            for i in range(num_layers)
        ))
        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for i in range(num_layers)
        ))
        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out


class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, attention=False, lstm_att=False, lstm=False,
                 upsample=False):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.att = attention
        self.lstm_att = lstm_att
        self.lstm = lstm
        if self.lstm_att:
            if self.att:
                self.lstm_layer = nn.LSTM(input_size=num_f_maps * 2, hidden_size=num_f_maps, batch_first=True)
            else:
                print("If LSTM layer desired turn on the attention flag")

        if attention:
            self.attention = nn.MultiheadAttention(num_f_maps, num_heads=5,
                                                   batch_first=True)  # num_f_maps must be divisible by num_heads
            self.layer_norm = nn.LayerNorm(num_f_maps)
        self.upsample = upsample
        if upsample:
            self.deconv = nn.ConvTranspose1d(num_f_maps, num_f_maps, kernel_size=3, stride=2, padding=1,
                                             output_padding=1)
            self.conv_out_refine = nn.Conv1d(num_f_maps, num_f_maps, 3, padding=1)
            self.conv_out_refine2 = nn.Conv1d(num_f_maps, num_f_maps, 3, padding=1)
            self.conv_out_refine3 = nn.Conv1d(num_f_maps, num_classes, 1)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)

        if self.upsample:
            out = self.deconv(out)
            out = self.relu(self.conv_out_refine(out))
            out = self.relu(self.conv_out_refine2(out))
            # out = self.conv_out_refine3(out)

        if self.att:
            # reshape for multi-head attention
            out_att = out.permute(0, 2, 1)  # [batch_size, sequence_length, feature_dimension]
            # apply multi-head attention
            out_att, _ = self.attention(out_att, out_att, out_att)
            # apply layer normalization
            out_att = self.layer_norm(out_att)
            # reshape back to original shape
            out_att = out_att.permute(0, 2, 1)  # [batch_size, feature_dimension, sequence_length]
            if self.lstm_att:
                out = torch.cat((out, out_att), dim=1)
                out = out.permute(0, 2, 1)
                out, _ = self.lstm_layer(out)  # apply LSTM layer
                out = self.conv_out(out.permute(0, 2, 1))  # (batch_size, num_classes, seq_length)
                return out
            else:
                out_att = self.conv_out(out_att)
                return self.conv_out(out_att)
        else:
            return self.conv_out(out)

        # out = self.conv_out(out)  # yields (batch_size, num_classes, seq_length)
        return out


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, split, sample_size=5,
                 model_type='mstcn2',
                 transformer_params={},
                 concat_kinematic_data=False, 
                 kinematic_features_path="/datashare/APAS/kinematics_npy",
				 weighted=False, class_weights=None,
				 lstm=False, lstm_att=False, ridge_reg=False, attention=False):
        """
        num_layers_PG: number of prediction generation layers
        num_layers_R: number of layers in refienment stage
        num_R: number of refienment stages
        num_f_maps: number of feature maps
        dim: features dimension
        num_classes: number of classes
        split: fold number
        sample_size: sample size for interpolation
        model_type: choose model type: mstcn2, asformer
		transformer_params: parameters for transformer model
        concat_kinematic_data: concatenating kinematic data to video data
        kinematic_features_path: path to kinematic features
		weighted: weighted loss
        class_weights
        lstm: add lstm layer or not
        lstm_att: MSTCN++ with attention on each refinment prediction, concatenated with the original prediction and passing to LSTM layer
        ridge_reg: add ridge regression
        attention: Apply attention mechanism on each refinement prediction
        """  
        # Choose the desired model
        if model_type == 'mstcn2':
            if lstm_att:
                self.model = MS_TCN2_Modified(num_layers_PG=num_layers_PG, num_layers_R=num_layers_R, num_R=num_R, 
                num_f_maps=num_f_maps, dim=dim, num_classes=num_classes, attention=attention, lstm_att=lstm_att, lstm=lstm)        	
            else:
                self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, weighted)

        else: #asformer
            self.model = MyTransformer(**transformer_params)
		
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.fold = split

        self.weighted = weighted
        self.weighted_str = "Weighted" if self.weighted else "Regular"

        self.attention = attention
        self.attention_flag = "Attention" if self.attention else "WithoutAttention"

        self.class_weights = class_weights is not None
        self.class_weights_str = "ClassWeighting" if self.class_weights else "NoClassWeighting"

        self.lstm = lstm
        self.lstm_flag = "LSTM_Attention_Refining" if self.lstm else "No_LSTM_Attention_Refining"

        self.lstm_att = lstm_att
        self.lstm_att_flag = "FinalGRU" if self.lstm_att else "FinalNoGRU"
        
        self.sample_size = sample_size
        self.sample_size_str = f"SampleSize{sample_size}"

        self.ridge_reg = ridge_reg
        self.ridge_reg_flag = "ridge_regularization" if self.ridge_reg else "Without_ridge_regularization"

        ext = [self.class_weights_str, self.sample_size_str, self.lstm_att_flag]
        self.exp_name = "-".join(ext)
        self.overlap = [.1, .25, .5]
        logger.add('logs/fold_' + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")
        self.concat_kinematic_data = concat_kinematic_data
        self.kinematic_features_path = kinematic_features_path

    def clear_ml_reporter(self, clogger, epoch, epoch_loss_train, batch_gen_train, epoch_loss_val, batch_gen_val,
                          correct_train, total_train, correct_val, total_val, edit_score_val, edit_score_train,
                          f1s_train, f1s_val):
        clogger.report_scalar(self.exp_name + " Losses - " + self.fold, "Train Loss", iteration=epoch + 1,
                              value=epoch_loss_train / len(batch_gen_train.list_of_examples))
        clogger.report_scalar(self.exp_name + " Losses - " + self.fold, "Validation Loss", iteration=epoch + 1,
                              value=epoch_loss_val / len(batch_gen_val.list_of_examples))
        clogger.report_scalar(self.exp_name + " Accuracies - " + self.fold, "Train Accuracy",
                              iteration=epoch + 1,
                              value=(float(correct_train) / total_train))
        clogger.report_scalar(self.exp_name + " Accuracies - " + self.fold, "Validation Accuracy",
                              iteration=epoch + 1,
                              value=(float(correct_val) / total_val))
        clogger.report_scalar("Validation Accuracies - " + self.fold, f"{self.sample_size} sample rate",
                              iteration=epoch + 1,
                              value=(float(correct_val) / total_val))
        clogger.report_scalar("Validation Edit Score - " + self.fold, f"{self.sample_size} sample rate",
                              iteration=epoch + 1,
                              value=np.mean(edit_score_val))
        clogger.report_scalar("Train Edit Score - " + self.fold, f"{self.sample_size} sample rate",
                              iteration=epoch + 1,
                              value=np.mean(edit_score_train))
        for k in range(len(self.overlap)):
            clogger.report_scalar(self.exp_name + " F1 Validation - " + self.fold,
                                  f"F1@{int(self.overlap[k] * 100)}",
                                  iteration=epoch + 1,
                                  value=(float(f1s_val[k]) / len(batch_gen_val.list_of_examples)))
            clogger.report_scalar(self.exp_name + " F1 Train - " + self.fold, f"F1@{int(self.overlap[k] * 100)}",
                                  iteration=epoch + 1,
                                  value=(float(f1s_train[k]) / len(batch_gen_train.list_of_examples)))

    def l2_regularization_loss(self, lambda_l2):
        l2_loss = 0.0
        for param in self.model.parameters():
            l2_loss += torch.norm(param, p=2) ** 2
        return 0.5 * lambda_l2 * l2_loss

    def train(self, save_dir, batch_gen_train, batch_gen_val, num_epochs, batch_size, learning_rate, device, clogger,
              lambda_l2):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        max_metric_F1, best_epoch = 0, 0
        for epoch in tqdm(range(num_epochs)):
            if self.weighted:
                for i, w in enumerate(self.model.Ws):
                    clogger.report_scalar(self.exp_name + "-" + self.fold + " Weights", f"Stage {i + 1}",
                                          iteration=epoch + 1,
                                          value=w[0].item())
            epoch_loss_train = 0
            correct_train = 0
            total_train = 0
            f1s_train = [0, 0, 0]
            edit_score_train = []
            self.model.train()
            batch_i = 0
            batch_loss = 0

            # Training loop
            while batch_gen_train.has_next():
                batch_i += 1
                batch_input, batch_target, mask = batch_gen_train.next_batch(batch_size, concat_kinematic_data=self.concat_kinematic_data)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)
                loss = 0
                for i, p in enumerate(predictions):
                    cur_pred_loss = 0
                    cur_pred_loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                             batch_target.view(-1))
                    cur_pred_loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                        min=0, max=16) * mask[:, :, 1:])
                    if self.weighted:
                        cur_pred_loss = cur_pred_loss * self.model.Ws[i]
                    loss += cur_pred_loss

                # Compute L2 regularization loss
                if self.ridge_reg:
                    l2_regularization_loss = self.l2_regularization_loss(lambda_l2)
                    # Compute total loss
                    loss += l2_regularization_loss

                ####################################################################################################
                # Calculate every couple of batches of size 1
                batch_loss += loss / predictions.shape[0]  # The loss is divided by the number of predictions
                epoch_loss_train += loss.item()
                if batch_i % 5 == 0:
                    batch_loss.backward()
                    optimizer.step()
                    batch_loss = 0
                ####################################################################################################

                _, predicted = torch.max(predictions[-1].data, 1)
                correct_train += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_train += torch.sum(mask[:, 0, :]).item()

                ####################################################################################################
                # Calculate metrics
                tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

                for s in range(len(self.overlap)):
                    tp1, fp1, fn1 = f_score(predicted.view(-1).tolist(), batch_target.view(-1).tolist(),
                                            self.overlap[s],
                                            train=True)
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                for s in range(len(self.overlap)):
                    precision = tp[s] / float(tp[s] + fp[s])
                    recall = tp[s] / float(tp[s] + fn[s])
                    f1 = 2.0 * (precision * recall) / (precision + recall)
                    f1 = np.nan_to_num(f1) * 100
                    f1s_train[s] += f1

                edit_score_train.append(edit_score(predicted.view(-1).tolist(), batch_target.view(-1).tolist()))
                ####################################################################################################

            ####################### Validation #######################
            epoch_loss_val = 0
            correct_val = 0
            total_val = 0
            f1s_val = [0, 0, 0]
            edit_score_val = []
            self.model.eval()
            while batch_gen_val.has_next():
                batch_input, batch_target, mask = batch_gen_val.next_batch(batch_size, concat_kinematic_data=self.concat_kinematic_data)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                predictions = self.model(batch_input)
                loss = 0
                for i, p in enumerate(predictions):
                    if self.weighted:
                        temp = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                       batch_target.view(-1))
                        temp += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:])
                        if self.attention:
                            temp += 0.5 * torch.mean(F.softmax(p[:, :, 1:], dim=1) * (
                                    F.log_softmax(p[:, :, 1:], dim=1) - F.log_softmax(p.detach()[:, :, :-1], dim=1)))
                        loss += temp * self.model.Ws[i]

                    else:
                        loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                        batch_target.view(-1))
                        loss += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:])
                        if self.attention:
                            loss += 0.5 * torch.mean(F.softmax(p[:, :, 1:], dim=1) * (
                                    F.log_softmax(p[:, :, 1:], dim=1) - F.log_softmax(p.detach()[:, :, :-1], dim=1)))

                epoch_loss_val += loss.item()
                _, predicted = torch.max(predictions[-1].data, 1)
                correct_val += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_val += torch.sum(mask[:, 0, :]).item()
                tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
                for s in range(len(self.overlap)):
                    tp1, fp1, fn1 = f_score(predicted.view(-1).tolist(), batch_target.view(-1).tolist(),
                                            self.overlap[s],
                                            train=True)
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
                for s in range(len(self.overlap)):
                    precision = tp[s] / float(tp[s] + fp[s])
                    recall = tp[s] / float(tp[s] + fn[s])
                    f1 = 2.0 * (precision * recall) / (precision + recall)
                    f1 = np.nan_to_num(f1) * 100
                    f1s_val[s] += f1
                edit_score_val.append(edit_score(predicted.view(-1).tolist(), batch_target.view(-1).tolist()))
            batch_gen_train.reset()
            batch_gen_val.reset()
            if (float(f1s_val[-1]) / len(batch_gen_val.list_of_examples)) > max_metric_F1:
                max_metric_F1 = (float(f1s_val[-1]) / len(batch_gen_val.list_of_examples))
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_dir + "/best.model")
                torch.save(optimizer.state_dict(), save_dir + "/best.opt")

            self.clear_ml_reporter(clogger, epoch, epoch_loss_train, batch_gen_train, epoch_loss_val, batch_gen_val,
                                   correct_train, total_train, correct_val, total_val, edit_score_val, edit_score_train,
                                   f1s_train, f1s_val)

            logger.info(
                "[epoch %d]: epoch loss train set = %f,   acc_train = %f" % (
                    epoch + 1, epoch_loss_train / len(batch_gen_train.list_of_examples),
                    float(correct_train) / total_train))
            logger.info(
                "[epoch %d]: epoch loss validation set = %f,   acc_val = %f" % (
                    epoch + 1, epoch_loss_val / len(batch_gen_val.list_of_examples),
                    float(correct_val) / total_val))

        logger.info(f"{self.exp_name} Run: Best Validation F1@50 = %f at epoch = %f" % (max_metric_F1, best_epoch + 1))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        """ Run infernce"""
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/best.model"))
            list_of_vids = vid_list_file
            correct_test = 0
            edit_score_test = 0
            for vid in list_of_vids:
                # print vid
                features = np.load(features_path + vid)
                if self.concat_kinematic_data:
                    kinematic_features = np.load(self.kinematic_features_path + '/' + vid)
                    kinematic_features_sampled = kinematic_features[:, ::sample_rate]
                    features_sampled = features[:, ::sample_rate]
                    num_frames = min(kinematic_features_sampled.shape[1], features_sampled.shape[1])
                    concat_features = np.vstack((features_sampled[:,0:num_frames], kinematic_features_sampled[:,0:num_frames]))
                    input_x = torch.tensor(concat_features, dtype=torch.float)
                else:
                    features = features[:, ::sample_rate]
                    input_x = torch.tensor(features, dtype=torch.float)
                
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

# ------------------- ASFormer ------------------------------
class MT_RNN_dp(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_classes_list, bidirectional, dropout, num_layers=2):
        super(MT_RNN_dp, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        else:
            raise NotImplemented
        # The linear layer that maps from hidden state space to tag space
        self.output_heads = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes_list[s])) for s in range(len(num_classes_list))])

    def forward(self, batch_input_kinematics):
        lengths = torch.tensor([batch_input_kinematics.shape[2]])
        outputs = []
        batch_input_kinematics = batch_input_kinematics.permute(0, 2, 1)
        batch_input_kinematics = self.dropout(batch_input_kinematics)

        packed_input = pack_padded_sequence(batch_input_kinematics, lengths=lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(packed_input)

        unpacked_rnn_out, unpacked_rnn_out_lengths = pad_packed_sequence(rnn_output, padding_value=-1, batch_first=True)
        # flat_X = torch.cat([unpacked_ltsm_out[i, :lengths[i], :] for i in range(len(lengths))])
        unpacked_rnn_out = self.dropout(unpacked_rnn_out)
        for output_head in self.output_heads:
            outputs.append(output_head(unpacked_rnn_out).permute(0, 2, 1))
        return outputs

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6)  # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type):  # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)

        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder', 'decoder']

        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()

    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)))
        for i in range(self.bl):
            window_mask[:, :, i:i + self.bl] = 1
        return window_mask.to(device)

    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder

        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _block_wise_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                                  torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)], dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,
                                                                                                     1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)

        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                                  torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)], dim=-1)

        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k,
                       torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v,
                       torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask,
                                  torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)

        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)],
                      dim=0)  # special case when self.bl = 1
        v = torch.cat([v[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)], dim=0)
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat(
            [padding_mask[:, :, i * self.bl:(i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)],
            dim=0)  # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask

        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
        #         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type,
                                  stage=stage)  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, max_len=10_000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)

    #         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]


class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in  # 2**i
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        """
        :param x: (N, C, L)
        :param mask:
        :return:
        """

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()  # self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in  # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p * idx_decoder)


class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, lstm_dropout):
        super(MyTransformer, self).__init__()
        self.rnn = MT_RNN_dp('LSTM', 36, 100, [6], dropout=lstm_dropout, num_layers=3, bidirectional=True).to(device)
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='sliding_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(num_decoders)])  # num_decoders

    def forward(self, batch_input):
        #batch_input = batch_input.transpose(1, 2)
        mask = torch.ones_like(batch_input, device=device)
        out, feature = self.encoder(batch_input, mask)
        outputs = out.unsqueeze(0)

        for i, decoder in enumerate(self.decoders):
            # probs = F.softmax(out, dim=1)
            if i == len(self.decoders):
                out, feature = decoder(F.softmax(out, dim=1), feature, mask)
            else:
                out, feature = decoder(F.softmax(out, dim=1), feature, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs
