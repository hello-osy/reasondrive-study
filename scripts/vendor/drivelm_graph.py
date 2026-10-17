"""DriveLM coordinate matching and graph filtering, extracted without algorithm changes.
Source: OpenDriveLab/DriveLM challenge/evaluation.py
Commit: 1de72a74b257e5373400fa68239e99bd5d20580a
Apache-2.0; see LICENSE-DriveLM. GPT calls and unused methods omitted.
"""
import re
import numpy as np


class DriveLMGraph:
    def eval_graph(self, question):
            # check if answer in self.graph  
            question_nums = re.findall(r'\d+\.\d+', question)
            question_nums = np.array([list(map(float, x.split()))[0] for x in question_nums]).reshape(-1, 2)
            question_nums = [list(i) for i in question_nums]
            for q in question_nums:
                if q not in self.graph:
                    return False
            return True

    def match_result(self, answer, GT):
            """
            answer: [[1.,2.], [2., 3.]]
            GT: [[1., 2.], [2., 3.]]
            """
            answer_nums = re.findall(r'\d+\.\d+', answer)
            GT_nums = re.findall(r'\d+\.\d+', GT)
            # transform string into float
            if len(answer_nums) % 2 != 0:
                answer_nums = answer_nums[:-1]
            answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
            GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)
            length = len(GT_nums)

            matched_out = []
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            for pred in answer_nums:
                closest_distance = float('inf')
                closest_gt = None
                closest_id = None
                for i, gt in enumerate(GT_nums):
                    distance = np.sum(np.abs(pred - gt))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_gt = gt
                        closest_id = i

                if closest_distance < 16:
                    true_positives += 1
                    matched_out.append(closest_gt)  
                    GT_nums = np.delete(GT_nums, closest_id, axis=0) 
                else:
                    false_positives += 1
                
            false_negatives = length - true_positives
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            F1 = 2 * precision * recall / (precision + recall + 1e-8)

            return matched_out, F1

    def set_graph(self, answer, GT):
            self.graph, _ = self.match_result(answer, GT)
            self.graph = [list(i) for i in self.graph]
