# Understanding and Benchmarking Zero-Shot Adversarial Robustness for Foundation Models (ICLR 2023)

<p align="center">
  <p align="center" margin-bottom="0px">
    <a href="http://www.cs.columbia.edu/~mcz/"><strong>Chengzhi Mao*</strong></a>
    ·
    <a href="https://www.scottgeng.com/"><strong>Scott Geng*</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~junfeng/"><strong>Junfeng Yang</strong></a>
    ·
    <a href="https://xinw.ai/"><strong>Xin Wang</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~vondrick/"><strong>Carl Vondrick</strong></a></p>
    <p align="center" margin-top="0px"><a href="https://arxiv.org/abs/2212.07016">https://arxiv.org/abs/2212.07016</a></p>
</p>

Pretrained vision-language foundation models like CLIP have exhibited strong generalization over unseen tasks, yet imperceptible adversarial perturbations can significantly reduce their performance. Given that such large-scale models are becoming a form of infrastructure in practice, understanding and defending their robustness has become an important new problem space. In particular, our recent work demonstrates that existing standard adversarial training techniques suffer from a catch-22 when applied to zero-shot models: without adversarial training, the model is vulnerable to attacks, but with adversarial training, the model loses its zero-shot capabilities. This problem is partially addressed by our introduced text-guided contrastive adversarial training loss, but a gap still remains. To spur further advances in this important space, we propose a defence challenge.

## Zero-Shot Adversarial Robustness Challenge

| Defence Method 	| Submitted By    	| Accuracy<br>(Robust) | Accuracy<br>(Clean) 	  | Submission Date 	|
|----------------	|-----------------	|----------------	|-----------------	|-----------------	|
|       <a href="https://cv.cs.columbia.edu/mcz/TeCoA/TeCoAmodel_best.pth.tar">TeCoA w/ Finetuning</a>        | (initial entry) 	|      **38.18%**  |         55.97%     |      Mar 1, 2023        |
|       TeCoA w/ Prompting        | (initial entry) 	|      **32.00%**  |         53.00%     |      Mar 1, 2023        |
|       Standard Adv. Training w/ Finetuning        | (initial entry) 	|      10.62%  |         18.49%     |      Mar 1, 2023        |
|   Vanilla CLIP-B/32 (no defence)  | (initial entry) 	|      6.57        |     **64.56%**     |      Mar 1, 2023        |


If you obtain better results, you are welcome to update your results here by sending email or push request.

## CLIP Model

### Environment setup:

install virtual environment:
`pip install virtualenv`

`virtualenv ZSRobust`

`source ZSRobust/venv/bin/activate`

`pip install -r requirement_final.txt`

Then replace the files in the replace folder to the source code in your environmet:  

replace `ZSRobust/venv/lib/python3.8/site-packages/clip/clip.py` and `ZSRobust/venv/lib/python3.8/site-packages/clip/model.py` with clip.py and model.py in the replace folder respectively. 
This is adapting CLIP to enable append token for prompting.

replace the `ZSRobust/venv/lib/python3.8/site-packages/torchvision/datasets` with the files in `replace/torchvision.datasets` 
for updated dataset loader

### Running

For adapting for zero-shot adversarial robustness with visual prompting, run

`python visual_prompt.py`

An example for running the eps=1 training is:
` CUDA_VISIBLE_DEVICES=4,5 python visual_prompt.py --batch_size 256 --root /local/vondrick/chengzhi --dataset ImageNet --add_prompt_size 100 --learning_rate 40 --name feimogu --train_eps 1 --train_numsteps 2 --train_stepsize 1`

For finetuning, run

`python finetuning.py`

An example for running the training: 
`CUDA_VISIBLE_DEVICES=4,5 python finetuning.py --batch_size 256 --root /local/vondrick/chengzhi --dataset ImageNet --name feimogu --train_eps 1 --train_numsteps 2 --train_stepsize 1`
