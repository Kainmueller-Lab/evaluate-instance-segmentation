import unittest
import numpy as np
import tempfile
import os
import h5py
import zarr

from evalinstseg.evaluate import evaluate_volume
from evalinstseg.match import instance_mask, greedy_many_to_many_matching
from evalinstseg.localize import compute_localization_criterion
from evalinstseg.util import check_and_fix_sizes, check_fix_and_unify_ids


class TestInstanceMask(unittest.TestCase):
    """Test the instance_mask function"""
    
    def test_single_channel(self):
        """Test instance_mask with single channel data"""
        labels = np.zeros((1, 10, 10), dtype=int)
        labels[0, 2:5, 2:5] = 1  # Instance 1 (no overlap)
        labels[0, 6:9, 6:9] = 2  # Instance 2 (no overlap)
        
        # Test getting instance 0 (should return instance 1)
        mask = instance_mask(labels, 0)
        expected = np.zeros((10, 10), dtype=bool)
        expected[2:5, 2:5] = True
        
        np.testing.assert_array_equal(mask, expected)
        
        # Test getting instance 1 (should return instance 2)
        mask = instance_mask(labels, 1)
        expected = np.zeros((10, 10), dtype=bool)
        expected[6:9, 6:9] = True
        
        np.testing.assert_array_equal(mask, expected)
    
    def test_multi_channel(self):
        """Test instance_mask with multi-channel data"""
        labels = np.zeros((3, 10, 10), dtype=int)
        labels[0, 2:6, 2:6] = 1  # Instance 1 in channel 0
        labels[1, 5:9, 5:9] = 2  # Instance 2 in channel 1
        labels[2, 3:7, 3:7] = 1  # Instance 1 also in channel 2
        
        # Test getting instance 0 (should return union of channels)
        mask = instance_mask(labels, 0)
        expected = np.zeros((10, 10), dtype=bool)
        expected[2:6, 2:6] = True  # From channel 0
        expected[3:7, 3:7] = True  # From channel 2
        
        np.testing.assert_array_equal(mask, expected)
    
    def test_nonexistent_instance(self):
        """Test instance_mask with non-existent instance"""
        labels = np.zeros((1, 10, 10), dtype=int)
        labels[0, 2:6, 2:6] = 1  # Only instance 1
        
        # Test getting instance 5 (should return all False)
        mask = instance_mask(labels, 5)
        expected = np.zeros((10, 10), dtype=bool)
        
        np.testing.assert_array_equal(mask, expected)


class TestGreedyMatching(unittest.TestCase):
    """Test the greedy many-to-many matching algorithm"""
    
    def test_perfect_matches(self):
        """Test with perfect 1:1 matches"""
        gt = np.zeros((2, 10, 10), dtype=int)
        gt[0, 2:6, 2:6] = 1  # GT instance 1
        gt[1, 5:9, 5:9] = 2  # GT instance 2
        
        pred = np.zeros((2, 10, 10), dtype=int)
        pred[0, 2:6, 2:6] = 1  # Perfect match for GT 1
        pred[1, 5:9, 5:9] = 2  # Perfect match for GT 2
        
        # Create perfect localization matrix
        locMat = np.zeros((3, 3))
        locMat[1, 1] = 1.0  # GT1-Pred1 perfect match
        locMat[2, 2] = 1.0  # GT2-Pred2 perfect match
        
        result = greedy_many_to_many_matching(gt, pred, locMat, 0.5)
        
        # Should have 2 matches
        self.assertEqual(len(result), 2)
        self.assertIn(0, result)  # GT instance 0 -> Pred instance 0
        self.assertIn(1, result)  # GT instance 1 -> Pred instance 1
        self.assertEqual(result[0], [0])
        self.assertEqual(result[1], [1])
    
    def test_no_matches(self):
        """Test with no matches above threshold"""
        gt = np.zeros((2, 10, 10), dtype=int)
        gt[0, 2:6, 2:6] = 1
        
        pred = np.zeros((2, 10, 10), dtype=int)
        pred[0, 7:9, 7:9] = 1  # No overlap with GT
        
        # Create low overlap matrix
        locMat = np.zeros((3, 3))
        locMat[1, 1] = 0.3  # Below threshold
        
        result = greedy_many_to_many_matching(gt, pred, locMat, 0.5)
        
        # Should return None for no matches
        self.assertIsNone(result)
    
    def test_many_to_one_matching(self):
        """Test many predictions matching one GT"""
        gt = np.zeros((1, 15, 15), dtype=int)
        gt[0, 3:6, 3:6] = 1   # blob A
        gt[0, 3:6, 9:12] = 1  # blob B

        # Pred: each blob as its own instance | essentially a false split
        pred = np.zeros((2, 15, 15), dtype=int)
        pred[0, 3:6, 3:6] = 1   # prediction for blob A
        pred[1, 3:6, 9:12] = 2  # prediction for blob B

        # Compute clDice-based localization matrix for 2 preds, 1 GT label
        locMat, _, _, _ = compute_localization_criterion(
            pred, gt,
            num_pred_labels=2,
            num_gt_labels=1,
            localization_criterion='cldice'
        )

        # both preds should have some overlap with GT
        self.assertGreater(locMat[1, 1], 0.0)
        self.assertGreater(locMat[1, 2], 0.0)

        # Use a low threshold so both parts qualify (each covers half the skeleton)
        thresh = 0.1
        result = greedy_many_to_many_matching(gt, pred, locMat, thresh)

        # We expect one GT entry (id 0 in the result dict)
        self.assertIsNotNone(result)
        self.assertIn(0, result)

        # And that GT should have both prediction instances assigned
        self.assertEqual(set(result[0]), {0, 1})
        self.assertEqual(len(result[0]), 2)

class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions"""
    
    def test_size_checking_crop(self):
        """Test that size checking and cropping works correctly"""
        # GT smaller than pred
        gt = np.zeros((1, 10, 10), dtype=int)
        pred = np.zeros((1, 15, 15), dtype=int)
        
        gt_cropped, pred_cropped = check_and_fix_sizes(gt, pred, 2)
        
        # Expect crop pred to match gt size
        self.assertEqual(gt_cropped.shape, (1, 10, 10))
        self.assertEqual(pred_cropped.shape, (1, 10, 10))
    
    def test_relabeling(self):
        """Test that relabeling works correctly"""
        gt = np.zeros((1, 10, 10), dtype=int)
        gt[0, 2:5, 2:5] = 1
        gt[0, 6:9, 6:9] = 2
        
        pred = np.zeros((1, 10, 10), dtype=int)
        pred[0, 2:5, 2:5] = 5  # Sparse labeling
        pred[0, 6:9, 6:9] = 10  # Sparse labeling
        
        gt_rel, pred_rel = check_fix_and_unify_ids(gt, pred, None, False)
        
        # Should have sequential labels
        self.assertEqual(set(np.unique(gt_rel)), {0, 1, 2})
        self.assertEqual(set(np.unique(pred_rel)), {0, 1, 2})


class TestLocalizationCriterion(unittest.TestCase):
    """Test localization criterion computation"""
    
    def test_iou_computation(self):
        """Test IoU computation"""
        gt = np.zeros((1, 10, 10), dtype=int)
        gt[0, 2:6, 2:6] = 1
        
        pred = np.zeros((1, 10, 10), dtype=int)
        pred[0, 3:7, 3:7] = 1  # Overlapping but not perfect
        
        locMat, recallMat, precMat, _ = compute_localization_criterion(
            pred, gt, 1, 1, 'iou'
        )
        
        # Should have reasonable IoU values
        self.assertGreater(locMat[1, 1], 0)  # Some overlap
        self.assertLess(locMat[1, 1], 1)     # Not perfect overlap
    
    def test_cldice_computation(self):
        """Test centerline dice computation"""
        gt = np.zeros((1, 10, 10), dtype=int)
        gt[0, 2:6, 2:6] = 1
        
        pred = np.zeros((1, 10, 10), dtype=int)
        pred[0, 3:7, 3:7] = 1
        
        locMat, recallMat, precMat, _ = compute_localization_criterion(
            pred, gt, 1, 1, 'cldice'
        )
        
        # Should have reasonable cldice values
        self.assertGreater(locMat[1, 1], 0)
        self.assertLessEqual(locMat[1, 1], 1)


class TestEndToEnd(unittest.TestCase):
    """Test complete evaluation pipeline"""
    
    def test_simple_evaluation(self):
        """Test complete evaluation on simple synthetic data"""
        # Create simple test case
        gt = np.zeros((1, 20, 20), dtype=int)
        gt[0, 2:8, 2:8] = 1
        gt[0, 12:18, 12:18] = 2
        
        pred = np.zeros((1, 20, 20), dtype=int)
        pred[0, 3:9, 3:9] = 1  # Slightly shifted
        pred[0, 11:17, 11:17] = 2  # Slightly shifted
        
        # Run evaluation
        result = evaluate_volume(
            gt, pred, 2, "test_output",
            localization_criterion="iou",
            assignment_strategy="greedy",
            add_general_metrics=["avg_gt_skel_coverage"],
            evaluate_false_labels=True
        )
        
        # Check basic metrics
        self.assertEqual(result.metricsDict["general"]["Num GT"], 2)
        self.assertEqual(result.metricsDict["general"]["Num Pred"], 2)
        self.assertGreater(result.metricsDict["general"]["TP_05"], 0)
    
    def test_real_data_format(self):
        """Test with data in the same format as real evaluation"""
        
        # Create test data in HDF5 and Zarr format
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create HDF5 prediction file
            pred_file = os.path.join(temp_dir, "test_pred.hdf")
            with h5py.File(pred_file, 'w') as f:
                pred_data = np.zeros((15, 1, 50, 50), dtype=int)
                for i in range(15):
                    pred_data[i, 0, 5+i:10+i, 5+i:10+i] = i+1
                f.create_dataset('vote_instances', data=pred_data)
            
            # Create Zarr ground truth file
            gt_file = os.path.join(temp_dir, "test_gt.zarr")
            gt_data = np.zeros((1, 50, 50), dtype=int)
            for i in range(15):
                gt_data[0, 5+i:10+i, 5+i:10+i] = i+1
            
            # Use newer zarr API
            root = zarr.open(gt_file, mode='w')
            volumes = root.create_group('volumes')
            volumes.create_dataset('gt_instances', data=gt_data, shape=gt_data.shape, dtype=gt_data.dtype)
            
            # Test that we can read the files
            pred_loaded = h5py.File(pred_file, 'r')['vote_instances'][:]
            gt_loaded = zarr.open(gt_file)['volumes/gt_instances'][:]
            
            self.assertEqual(pred_loaded.shape, (15, 1, 50, 50))
            self.assertEqual(gt_loaded.shape, (1, 50, 50))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestInstanceMask,
        TestGreedyMatching,
        TestDataPreprocessing,
        TestLocalizationCriterion,
        TestEndToEnd
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
