export CUDA_VISIBLE_DEVICES=1

#cd ..

for model in MCTEM
do

for preLen in 96 192 336 720
do

# ETTh1
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 3 \
  --channel_independence 1 \
  --temporal_independence 1 \
  --num_blocks 1 \
  --num_layers 4 \

done

