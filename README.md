# Neural Sign Language Translation

This repo contains the training and evalution code of Sign2Text setup for translation sign langauge videos to spoken language sentences. 

This code is based on [an earlier version of Luong et al.'s Neural Machine Translation Tutorial](https://github.com/tensorflow/nmt/tree/tf-1.2). 

## Requirements
* Download and extract [RWTH-PHOENIX-Weather 2014 T: Parallel Corpus of Sign Language Video, Gloss and Translation](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and then resize the images to 227x227
* Download and install Tensorflow 1.3.0+ 
* Download [AlexNet TensorFlow weights](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put it under the folder BaseModel
* Python 2.7

## Usage

### Training Sample Usage (From nslt folder)
python -m nmt --src=sign --tgt=de --train_prefix=../Data/phoenix2014T.train --dev_prefix=../Data/phoenix2014T.dev --test_prefix=../Data/phoenix2014T.test --out_dir=<your_output_dir> --vocab_prefix=../Data/phoenix2014T.vocab --source_reverse=True --num_units=1000 --num_layers=4 --num_train_steps=150000 --residual=True --attention=luong --base_gpu=<gpu_id> --unit_type=gru 

### Inference Sample Usage

python -m nmt --out_dir=<your_model_dir> --inference_input_file=<input_video_paths.sign> --inference_output_file=<predictions.de> --inference_ref_file=<ground_truth.de> --base_gpu=<gpu_id>


## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{camgoz2018neural,
      author = {Necati Cihan Camgoz and Simon Hadfield and Oscar Koller and Hermann Ney and Richard Bowden},
      title = {Neural Sign Language Translation},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2018}
    }
