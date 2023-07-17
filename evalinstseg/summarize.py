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
