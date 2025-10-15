import csv
from functools import reduce

import numpy as np


def deep_get(dictionary, key, default=None):
    """get value with key from nested dictionary

    Example:
    dictionary = {'confusion_matrix': {th_0_5: 0.1234}}
    key = 'confusion_matrix.th_0_5'
    v = deep_get(dictionary, key)
    => v = 0.1234

    Args
    ----
    dictionary : dict
        nested dictionary
    key: str
        string with segments separated by '.', each segment corresponds to
        on level in nested dict

    Returns
    -------
    value: extracts value for key from dictionary
    """
    return reduce(
        lambda d, k: d.get(k, default) if isinstance(d, dict) else
        default, key.split("."), dictionary
    )

def summarize_metric_dict(metric_dicts, samples, metrics, output_name, agg_inst_dict=None):
    """computes summary from dicts containing metrics and stores in csv

    Note
    ----
    writes csv file

    Args
    ----
    metric_dicts: list of dict
        list of computed metrics, one dict per sample
    samples: list of str
        list of samples for which metrics have been computed
    metrics: list of str
        list of metrics that should be written to file
    output_name: str
        name of csv file to be written
    agg_inst_dict: dict
        additional dict whose metrics will be written to file,
        used for writing metrics computed across samples
        (instead of per sample)
    """
    csvf = open(output_name, 'w', newline='')
    writer = csv.writer(csvf, delimiter=';',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    header = ['sample'] + [m.split('.')[-2] + ' ' + m.split('.')[-1] for m in
                      metrics]
    writer.writerow(header)

    summary = np.zeros((len(metric_dicts), len(metrics)))
    for i, (sample, metric_dict) in enumerate(zip(samples, metric_dicts)):
        for k in range(len(metrics)):
            v = deep_get(metric_dict, metrics[k])
            if v is not None:
                summary[i, k] = float(v)
            else:
                summary[i, k] = 0.0

        writer.writerow([sample] + list(summary[i]))
    writer.writerow(['mean'] + list(np.mean(summary, axis=0)))
    writer.writerow(['sum'] + list(np.sum(summary, axis=0)))

    # write average over instances
    if agg_inst_dict is not None:
        avg_inst = []
        for m in metrics:
            v = deep_get(agg_inst_dict, m)
            if v is not None:
                avg_inst.append(v)
            else:
                avg_inst.append(0.0)
        writer.writerow(['avg_inst'] + avg_inst)

    csvf.close()


def average_flylight_score_over_instances(samples_foldn, result):
    # heads up: hard coded for 0.5 average F1 + 0.5 average gt coverage
    threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fscores = []
    gt_covs = []
    tp = {}
    fp = {}
    fn = {}
    fm = []
    fs = []
    num_gt = []
    num_pred = []
    tp_05 = []
    tp_05_cldice = []
    gt_dim = []
    gt_covs_dim = []
    tp_05_dim = []
    gt_ovlp = []
    gt_covs_ovlp = []
    tp_05_ovlp = []

    for thresh in threshs:
        tp[thresh] = []
        fp[thresh] = []
        fn[thresh] = []
    for s in samples_foldn:
        # TODO: move type conversion to evaluate_file
        c_gen = result[s]["general"]
        gt_covs += list(np.array(
            c_gen["gt_skel_coverage"], dtype=np.float32))
        num_gt.append(c_gen["Num GT"])
        num_pred.append(c_gen["Num Pred"])
        if "FM" in c_gen.keys():
            fm.append(c_gen["FM"])
            fs.append(c_gen["FS"])
            tp_05.append(c_gen["TP_05"])
            tp_05_cldice += list(np.array(
                c_gen["TP_05_cldice"], dtype=np.float32))
        if "GT_dim" in c_gen.keys():
            gt_dim.append(c_gen["GT_dim"])
            tp_05_dim.append(c_gen["TP_05_dim"])
            gt_covs_dim += list(np.array(c_gen["gt_covs_dim"], dtype=np.float32))
        if "GT_overlap" in c_gen.keys():
            gt_ovlp.append(c_gen["GT_overlap"])
            tp_05_ovlp.append(c_gen["TP_05_overlap"])
            gt_covs_ovlp += list(np.array(c_gen["gt_covs_overlap"], dtype=np.float32))

        for thresh in threshs:
            tp[thresh].append(result[s][
                                  "confusion_matrix"][
                                  "th_" + str(thresh).replace(".", "_")][
                                  "AP_TP"])
            fp[thresh].append(result[s][
                                  "confusion_matrix"][
                                  "th_" + str(thresh).replace(".", "_")][
                                  "AP_FP"])
            fn[thresh].append(result[s][
                                  "confusion_matrix"][
                                  "th_" + str(thresh).replace(".", "_")][
                                  "AP_FN"])
    for thresh in threshs:
        fscores.append(2 * np.sum(tp[thresh]) / (
            2 * np.sum(tp[thresh]) + np.sum(fp[thresh]) + np.sum(fn[thresh])))
    avS = 0.5 * np.mean(fscores) + 0.5 * np.mean(gt_covs)

    per_instance_counts = {}
    per_instance_counts["general"] = {
        "Num GT": np.sum(num_gt),
        "Num Pred": np.sum(num_pred),
        "avg_gt_skel_coverage": np.mean(gt_covs),
        "avg_f1_cov_score": avS,
        "avFscore": np.mean(fscores),
        "FM": np.sum(fm),
        "FS": np.sum(fs),
        "TP_05": np.sum(tp_05),
        "TP_05_rel": np.sum(tp_05) / float(np.sum(num_gt)),
        "TP_05_cldice": tp_05_cldice,
        "avg_TP_05_cldice": np.mean(tp_05_cldice) if np.sum(tp_05) > 0 else 0.0,
        "GT_dim": np.sum(gt_dim),
        "TP_05_dim": np.sum(tp_05_dim),
        "TP_05_rel_dim": np.sum(tp_05_dim) / float(np.sum(gt_dim)),
        "avg_gt_cov_dim": np.mean(gt_covs_dim),
        "GT_overlap": np.sum(gt_ovlp),
        "TP_05_overlap": np.sum(tp_05_ovlp),
        "TP_05_rel_overlap": np.sum(tp_05_ovlp) / float(np.sum(gt_ovlp)),
        "avg_gt_cov_overlap": np.mean(gt_covs_ovlp)
    }
    per_instance_counts["confusion_matrix"] = {"avFscore": np.mean(fscores)}
    per_instance_counts["gt_covs"] = gt_covs
    per_instance_counts["tp"] = []
    per_instance_counts["fp"] = []
    per_instance_counts["fn"] = []
    for i, thresh in enumerate(threshs):
        per_instance_counts["tp"].append(np.sum(tp[thresh]))
        per_instance_counts["fp"].append(np.sum(fp[thresh]))
        per_instance_counts["fn"].append(np.sum(fn[thresh]))
        per_instance_counts["confusion_matrix"][
            "th_" + str(thresh).replace(".", "_")] = {
            "fscore": fscores[i]}
        if thresh == 0.5:
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_TP"] = np.sum(
                tp[thresh])
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_FP"] = np.sum(
                fp[0.5])
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_FN"] = np.sum(
                fn[0.5])

    return avS, per_instance_counts


def average_sets(acc_a, dict_a, acc_b, dict_b):
    threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc = np.mean([acc_a, acc_b])
    fscore = np.mean([dict_a["general"]["avFscore"],
        dict_b["general"]["avFscore"]])
    gt_covs = list(dict_a["gt_covs"]) + list(dict_b["gt_covs"])
    num_gt = dict_a["general"]["Num GT"] + dict_b["general"]["Num GT"]
    tp_05 = dict_a["general"]["TP_05"] + dict_b["general"]["TP_05"]
    tp_05_cldice = list(dict_a["general"]["tp_05_cldice"]) + \
            list(dict_b["general"]["tp_05_cldice"])

    per_instance_counts = {}
    per_instance_counts["general"] = {
        "Num GT": num_gt,
        "Num Pred": dict_a["general"]["Num Pred"] + dict_b["general"]["Num Pred"],
        "avg_gt_skel_coverage": np.mean([
            dict_a["general"]["avg_gt_skel_coverage"],
            dict_b["general"]["avg_gt_skel_coverage"]]),
        "avg_f1_cov_score": acc,
        "avFscore": fscore,
        "FM": dict_a["general"]["FM"] + dict_b["general"]["FM"],
        "FS": dict_a["general"]["FS"] + dict_b["general"]["FS"],
        "TP_05": tp_05,
        "TP_05_rel": tp_05 / float(num_gt),
        "avg_TP_05_cldice": np.mean(tp_05_cldice)
    }
    per_instance_counts["confusion_matrix"] = {"avFscore": fscore}
    per_instance_counts["gt_covs"] = gt_covs
    for i, thresh in enumerate(threshs):
        cm_a = dict_a["confusion_matrix"]["th_" + str(thresh).replace(".", "_")]
        cm_b = dict_b["confusion_matrix"]["th_" + str(thresh).replace(".", "_")]
        per_instance_counts["confusion_matrix"][
            "th_" + str(thresh).replace(".", "_")] = {
            "fscore": np.mean([cm_a["fscore"],cm_b["fscore"]])}
        if thresh == 0.5:
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_TP"] = \
                    cm_a["AP_TP"] + cm_b["AP_TP"]
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_FP"] = \
                    cm_a["AP_FP"] + cm_b["AP_FP"]
            per_instance_counts["confusion_matrix"]["th_0_5"]["AP_FN"] = \
                    cm_a["AP_FN"] + cm_b["AP_FN"]
    return acc, per_instance_counts

