import os
import unittest

import numpy as np

from evalinstseg import evaluate_volume


# work in progress
class TestMetrics(unittest.TestCase):
    # how to name stuff, convex and tubular instead of nuclei and neuron?
    def set_expected(self, num_gt, num_pred, avg_gt_cov, avg_f1_cov, gt_cov,
            avAP59, avAP19, avFscore59, avFscore19,
            tp_0_5, fp_0_5, fn_0_5, fs_0_5, fm_0_5):
        # define metric dictionary structure for expected values
        expected = {
                "general": {},
                "confusion_matrix": {"th_0_5":{}}
                }
        # set values
        expected["general"]["Num GT"] = num_gt
        expected["general"]["Num Pred"] = num_pred
        expected["general"]["avg_gt_skel_coverage"] = avg_gt_cov
        expected["general"]["avg_f1_cov_score"] = avg_f1_cov
        expected["general"]["gt_skel_coverage"] = gt_cov
        expected["confusion_matrix"]["avAP59"] = avAP59
        expected["confusion_matrix"]["avAP19"] = avAP19
        expected["confusion_matrix"]["avFscore59"] = avFscore59
        expected["confusion_matrix"]["avFscore19"] = avFscore19
        expected["confusion_matrix"]["th_0_5"]["AP_TP"] = tp_0_5
        expected["confusion_matrix"]["th_0_5"]["AP_FP"] = fp_0_5
        expected["confusion_matrix"]["th_0_5"]["AP_FN"] = fn_0_5
        expected["confusion_matrix"]["th_0_5"]["false_split"] = fs_0_5
        expected["confusion_matrix"]["th_0_5"]["false_merge"] = fm_0_5

        return expected


    def check_results(self, results, expected):
        # check general
        res = results["general"]
        exp = expected["general"]
        self.assertEqual(res["Num GT"], exp["Num GT"])
        self.assertEqual(res["Num Pred"], exp["Num Pred"])
        self.assertAlmostEqual(
            res["avg_gt_skel_coverage"],exp["avg_gt_skel_coverage"], 4)
        self.assertAlmostEqual(res["avg_f1_cov_score"], exp["avg_f1_cov_score"], 4)
        self.assertEqual(len(res["gt_skel_coverage"]), len(exp["gt_skel_coverage"]))
        for r, e in zip (res["gt_skel_coverage"], exp["gt_skel_coverage"]):
            self.assertAlmostEqual(r, e, 4)
        # check confusion table
        res = results["confusion_matrix"]
        exp = expected["confusion_matrix"]
        self.assertAlmostEqual(res["avAP59"], exp["avAP59"], 4)
        self.assertAlmostEqual(res["avAP19"], exp["avAP19"], 4)
        self.assertAlmostEqual(res["avFscore59"], exp["avFscore59"], 4)
        self.assertAlmostEqual(res["avFscore19"], exp["avFscore19"], 4)
        # check error quantities for confusion table at threshold 0.5
        res = results["confusion_matrix"]["th_0_5"]
        exp = expected["confusion_matrix"]["th_0_5"]
        self.assertEqual(res["AP_TP"], exp["AP_TP"])
        self.assertEqual(res["AP_FP"], exp["AP_FP"])
        self.assertEqual(res["AP_FN"], exp["AP_FN"])
        self.assertEqual(res["false_split"], exp["false_split"])
        self.assertEqual(res["false_merge"], exp["false_merge"])


    def run_test_case(self, config, gt, pred, expected):

        result_dict = evaluate_volume(
            gt,
            pred,
            config['ndim'],
            config["outFn"],
            config["localization_criterion"],
            config["assignment_strategy"],
            config["evaluate_false_labels"],
            config["add_general_metrics"],
            config["visualize"],
            config["visualize_type"],
            config["overlapping_inst"],
            config["partly"])
        print(result_dict)
        self.check_results(result_dict.metricsDict, expected)

    def test_2d_nuclei(self):
        print("todo: test 2d nuclei")


    def test_3d_nuclei(self):
        print("todo: test 2d nuclei")


    def test_3d_neuron(self):
        gt = np.zeros((2, 30, 30, 30), dtype=np.int32)
        gt[0, 14:17, 14:17, 5:25] = 1
        gt[1, 14:17, 5:25, 14:17] = 2

        print(np.sum(gt==1), np.sum(gt==2), np.sum(np.sum(gt>0, axis=0) > 1))

        # set parameters
        config = {
                "ndim": 3,
                "outFn": None,
                "localization_criterion": "cldice",
                "assignment_strategy": "greedy",
                "add_general_metrics": [
                    "avg_gt_skel_coverage",
                    "avg_f1_cov_score",
                    "false_merge",
                    "false_split"
                    ],
                "evaluate_false_labels": True,
                "visualize": False,
                "visualize_type": None,
                "overlapping_inst": True,
                "partly": False
                }

        # test case 1: perfect segmentation
        # (1.1) pred + gt overlaps
        pred = gt.copy()
        # set_expected: num_gt, num_pred, avg_gt_cov, avg_f1_cov, gt_cov,
        #   avAP59, avAP19, avFscore59, avFscore19,
        #   tp_0_5, fp_0_5, fn_0_5, fs_0_5, fm_0_5):
        self.run_test_case(config, gt, pred,
                self.set_expected(2, 2, 1.0, 1.0, [1.0, 1.0],
                    1.0, 1.0, 1.0, 1.0,
                    2, 0, 0, 0, 0)
                )

        # (1.2) gt overlaps
        pred = np.max(pred, axis=0)
        config["overlapping_inst"] = False
        # set expected values
        gt_cov = np.array([15/18.0, 1.0], dtype=np.float32)
        avg_gt_cov = np.mean(gt_cov)
        avg_f1_cov = np.mean([avg_gt_cov, 1.0])
        self.run_test_case(config, gt, pred,
                self.set_expected(2, 2, avg_gt_cov, avg_f1_cov,
                    gt_cov,
                    0.925, 1.0, 0.95, 1.0,
                    2, 0, 0, 0, 0)
                )

        # (1.3) no overlaps
        gt = np.max(gt, axis=0)
        self.run_test_case(config, gt, pred,
                self.set_expected(2, 2, 1.0, 1.0, [1.0, 1.0],
                    1.0, 1.0, 1.0, 1.0,
                    2, 0, 0, 0, 0)
                )

        # test case 2: erroneous segmentation
        gt = np.zeros((3, 30, 30, 30), dtype=np.int32)
        gt[0, 14:17, 14:17, 5:25] = 1
        gt[1, 14:17, 5:25, 14:17] = 2
        gt[2, 5:8, 5:25, 20:23] = 3

        pred = np.zeros((5, 30, 30, 30), dtype=np.int32)
        pred[0, 14:17, 14:17, 5:25] = 1
        pred[0, 14:17, 5:20, 14:17] = 1
        pred[1, 25:30, 25:30, 25:30] = 2
        pred[2, 5:8, 5:11, 20:23] = 3
        pred[3, 5:8, 19:22, 20:23] = 4
        pred[4, 1:5, 1:5, 1:5] = 5

        # (2.1) pred + gt overlaps
        config["overlapping_inst"] = False
        # gt_cov = np.array([1.0, 14/18.0, 8/18.0], dtype=np.float32)
        gt_cov = np.array([1.0, 0.0, 8/18.0], dtype=np.float32)
        avg_gt_cov = np.mean(gt_cov)
        ap19 = np.mean([4/15.0,] * 4 + [1/15.0,] * 3 + [0.0, 0.0])
        ap59 = np.mean([1/15.0,] * 6 + [0.0,] * 4)
        fscore19 = np.mean([0.5,] * 4 + [0.25,] * 3 + [0.0, 0.0])
        fscore59 = np.mean([0.25,] * 6 + [0.0,] * 4)
        avg_f1_cov = np.mean([avg_gt_cov, fscore19])
        self.run_test_case(config, gt, pred,
                self.set_expected(3, 5, avg_gt_cov, avg_f1_cov,
                    gt_cov,
                    ap59, ap19, fscore59, fscore19,
                    1, 4, 2, 2, 1)
                )


if __name__ == '__main__':
    unittest.main()

