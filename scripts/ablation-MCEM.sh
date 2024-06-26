export CUDA_VISIBLE_DEVICES=0

#cd ..

for model in MCTEM
do

for preLen in 96 192 336 720
do

# # ETT m1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./datasets/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1 \
#   --model $model \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 3 \
#   --channel_independence 1 \
#   --temporal_independence 0 \
#   --num_blocks 1 \
#   --num_layers 4 \



# # ETTh1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./datasets/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1 \
#   --model $model \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 3 \
#   --channel_independence 1 \
#   --temporal_independence 0 \
#   --num_blocks 1 \
#   --num_layers 4 \

# # ETTm2
# python -u run.py \
#   --is_training 1 \
#   --root_path ./datasets/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id ETTm2 \
#   --model $model \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 3 \
#   --channel_independence 1 \
#   --temporal_independence 0 \
#   --num_blocks 1 \
#   --num_layers 4 \

# ETTh2
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2 \
  --model $model \
  --data ETTh2 \
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
  --temporal_independence 0 \
  --num_blocks 1 \
  --num_layers 4 \

## electricity
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL \
  --model $model \
  --data electricity \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --channel_independence 1 \
  --temporal_independence 0 \
  --num_blocks 1 \
  --num_layers 4 \

# exchange
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange \
  --model $model \
  --data exchange_rate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 3 \
  --channel_independence 1 \
  --temporal_independence 0 \
  --num_blocks 1 \
  --num_layers 4 \

# traffic
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic \
  --model $model \
  --data traffic \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 3 \
  --channel_independence 1 \
  --temporal_independence 0 \
  --num_blocks 1 \
  --num_layers 4 \

# weather
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/weather/ \
  --data_path weather.csv \
  --model_id weather \
  --model $model \
  --data weather \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 3 \
  --channel_independence 1 \
  --temporal_independence 0 \
  --num_blocks 1 \
  --num_layers 4 \
    
done


# for preLen in 24 36 48 60
# do
# illness
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/illness/ \
#  --data_path national_illness.csv \
#  --model_id ili \
#  --model $model \
#  --data national_illness \
#  --features M \
#  --seq_len 36 \
#  --label_len 18 \
#  --pred_len $preLen \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 3
# done

done

