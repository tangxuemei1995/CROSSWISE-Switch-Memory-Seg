mkdir logs

# train
python smseg_main.py --do_train --train_data_path=./sample_data/train_mix.tsv --use_dict --classifier --use_memory --switch=soft_switch --eval_data_path=./sample_data/test.tsv --use_bert --bert_model=./bert-chinese  --decoder=crf  --max_seq_length=160 --max_ngram_size=160 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=1 --warmup_proportion=0.1 --learning_rate=2e-5 --ngram_num_threshold=2  --ngram_flag=av --av_threshold=2 --model_name=sample_model --model_set=test

# test
# python smseg_main.py --do_test --use_dict --classifier --switch=soft_switch --decoder=crf  --use_memory --train_data_path=./sample_data/train_mix.tsv --eval_data_path=./sample_data/test.tsv --test_data_path=./sample_data/test.tsv --eval_model=./new_model/sample_model_test/model.pt

