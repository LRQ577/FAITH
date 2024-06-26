export CUDA_VISIBLE_DEVICES=0

#cd ..

model= MCTEM

python -u run.py \
  --is_training 1 \
  --root_path ./datasets/for_short/ \
  --data_path traffic.csv \
  --model_id traffic \
  --model $model \
  --data traffic \
  --features M \
  --seq_len 12 \
  --label_len 12 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 \
  --channel_independence 1 \
  --temporal_independence 1 \
  --num_blocks 1 \
  --num_layers 4 \





