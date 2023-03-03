# Understanding and Benchmarking Zero-Shot Adversarial Robustness for Foundation Models

Pretrained large-scale vision-language models like CLIP have exhibited strong generalization over unseen tasks, yet imperceptible adversarial perturbations can significantly reduce their performance. Standard 


## Zero-Shot Adversarial Robustness Challenge

| Defence Method 	| Submitted By    	| Accuracy<br>(Robust) | Accuracy<br>(Clean) 	  | Submission Date 	|
|----------------	|-----------------	|----------------	|-----------------	|-----------------	|
|       TeCoA with Finetuning        | (initial entry) 	|      **38.18%**  |         55.97%     |      Mar 1, 2023        |
|   Vanilla CLIP-B/32 (no defence)  | (initial entry) 	|      6.57        |     **64.56%**     |      Mar 1, 2023        |


## CLIP Model

For adapting for zero-shot adversarial robustness with visual prompting, run

`python visual_prompt.py`


For finetuning, run

`python finetuning.py`

