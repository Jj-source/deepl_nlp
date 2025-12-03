- inference.ipynb

file that loads the model and runs inference

- bidirectional_gru_attn.ipynb

original notebook with code to train the model, which is an encoder decoder with 2 layers, with 
GRU as core recurrent architecture (chosen for faster training with, technically, only marginal efficiency loss),
a bidirectional decoder and attention.

previous iteration available in the previous iteration folders describe intermediate steps taken
prior to arriving to this model with also .txt files regarding train and test metrics

note: loaded model scores were slightly lower than those reached during training
originals can be found in the notebook before running it and in the 
`results/gru_bi_2layers_attn" files