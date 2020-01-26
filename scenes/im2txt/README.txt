

This repo contains a copy of im2txt (https://github.com/tensorflow/models/tree/master/im2txt),  pretrained models (https://github.com/tensorflow/models/issues/466) 

TF version: 1.15
Python: 3.7
I modified the code to work with python 3 and to run the code with no required command line arguments. Follow these steps to avoid possible errors.
STEPS:

   1- Download the pre-trained model checkpoints from the link above(e.g fine-tuned) and put them in a directory named: chkpoint_temp 
download words_count.txt
   2- Run the script rename_lstm_celss -to replace paths for checkpoint files correctly (e.g. ./model.ckpt-2000000 if the files are in current directory).
    run inference, example:
    python3 im2txt/run_inference.py --checkpoint_path=models/model.ckpt-2000000 --vocab_file=models/word_counts.txt --input_files images/image1.jpg
If you don't want to use CMD: modify the path in run_inference.py to your own path; Line 34, 37, & 38
3-Line 182 in caption_generator.py of im2txt should be:
most_likely_words = np.argsort(word_probabilities)[::-1]

4-run run_inference.py , and have fun!


