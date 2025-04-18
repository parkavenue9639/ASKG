CUDA_VISIBLE_DEVICES=1 python3 cnn_image_sentence_train.py \
--num_medterm 50176 \
--num_termclass 229 \
--train_batch_size 16 \
--eval_batch_size 16 \
--max_epochs 15 \
--num_disease 14 \
--data_dir ../data \
--checkpoint_path /Users/luchong/PycharmProjects/COV_CTR/COV-CTR/ASKG/report/log_cn_image_precise_encoder \
--encoder_path /Users/luchong/PycharmProjects/COV_CTR/COV-CTR/ASKG/report/log_cn_encoder \
--learning_rate 5e-5 \
--cnn_learning_rate 1e-4 \
--learning_rate_decay_start 0