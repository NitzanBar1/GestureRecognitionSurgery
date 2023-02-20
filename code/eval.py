#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
import os

gestures = {f"G{i}": i for i in range(6)}


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def custom_get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    for gt in frame_wise_labels:
        gt_split = gt.split()
        labels.append(gt_split[-1])
        starts.append(int(gt_split[0]))
        ends.append(int(gt_split[1]) + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float64)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, test=False, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    if test:
        Y, _, _ = custom_get_labels_start_end_time(ground_truth, bg_class)
        #Y = [gestures[row] for row in Y]
    else:
        Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, train=False, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    if train:
        y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
    else:
        y_label, y_start, y_end = custom_get_labels_start_end_time(ground_truth, bg_class)
        #y_label = [gestures[row] for row in y_label]

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')
    parser.add_argument('--results_path')
    parser.add_argument('--exp_name')
    args = parser.parse_args()

    ground_truth_path = '/datashare/APAS/transcriptions_gestures/'
    results_path = args.results_path
    exp_name = args.exp_name
    overlap = [.1, .25, .5]

    # Calculate metrics per fold
    for fold_id in range(5): 
        print("----------------------------------------")
        correct = 0
        total = 0
        edit = 0
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        list_of_videos = os.listdir(os.path.join(results_path, 'fold' + str(fold_id), exp_name))
        for vid in list_of_videos:
            gt_file = ground_truth_path + vid
            gt_content = read_file(gt_file + ".txt").split('\n')[0:-1]

            res_file = os.path.join(results_path, 'fold' + str(fold_id), exp_name, vid.split('.')[0])
            res_content = read_file(res_file).split('\n')[1].split()

            # accuracy calc
            for i in range(len(gt_content)):
                start_time = int(gt_content[i].split(' ')[0])
                end_time = int(gt_content[i].split(' ')[1])
                for j in range(start_time, end_time):
                    total += 1
                    if gt_content[i].split(' ')[-1] == res_content[j if j < len(res_content) else -1]:
                        correct += 1

            # edit score calc
            edit += edit_score(res_content, gt_content, test=True)

            # precision & recall calc
            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(res_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1

        # fold metrics summary
        acc = (100 * float(correct) / total)
        edit = ((1.0 * edit) / len(list_of_videos))
        print("Fold ", fold_id, " Accuracy: %.4f" %acc)
        print("Fold ", fold_id, " Edit Score: %.4f" %edit)


        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = np.nan_to_num(f1) * 100
            print("Fold ", fold_id,  " F1@%0.2f: %.4f" %(overlap[s], f1))


if __name__ == '__main__':
    main()