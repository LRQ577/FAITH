export CUDA_VISIBLE_DEVICES=0

#cd ..

model=MCTEM


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./datasets/for_short/ \
  --data_path solar.csv \
  --model_id solar \
  --model $model \
  --data solar \
  --features M \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 593 \
  --dec_in 593 \
  --c_out 593 \
  --des 'Exp' \
  --d_model 256 \
  --itr 1 \
  --channel_independence 1 \
  --temporal_independence 1 \
  --num_blocks 1 \
  --num_layers 1 \
  

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./datasets/for_short/ \
  --data_path covid.csv \
  --model_id covid \
  --model $model \
  --data covid \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 55 \
  --dec_in 55 \
  --c_out 55 \
  --des 'Exp' \
  --d_model 400 \
  --itr 1 \
  --channel_independence 1 \
  --temporal_independence 1 \
  --num_blocks 1 \
  --num_layers 1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id electricity \
  --model $model \
  --data electricity \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 32 \
  --itr 1 \
  --channel_independence 1 \
  --temporal_independence 1 \
  --num_blocks 1 \
  --num_layers 1 \
  --batch_size 128 \
  --learning_rate 0.005 \
  

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./datasets/for_short/ \
  --data_path traffic.csv \
  --model_id traffic \
  --model $model \
  --data traffic \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 32 \
  --itr 1 \
  --channel_independence 1 \
  --temporal_independence 1 \
  --num_blocks 1 \
  --num_layers 1 \
  --batch_size 32 \
  --learning_rate 0.005 \
  