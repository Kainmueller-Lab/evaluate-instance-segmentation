import numpy as np
import csv
from functools import reduce

def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else
        default, keys.split("."), dictionary
    )

def summarize_metric_dict(metric_dicts, names, metrics, output_name):

    csvf = open(output_name, 'w', newline='')
    writer = csv.writer(csvf, delimiter=';',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    header = ['sample'] + [m.split('.')[-2] + ' ' + m.split('.')[-1] for m in
                      metrics]
    writer.writerow(header)

    summary = np.zeros((len(metric_dicts), len(metrics)))
    for i, (name, metric_dict) in enumerate(zip(names, metric_dicts)):
        for k in range(len(metrics)):
            summary[i, k] = deep_get(metric_dict, metrics[k])

        writer.writerow([name] + list(summary[i]))

    writer.writerow(['mean'] + list(np.mean(summary, axis=0)))

    csvf.close()
