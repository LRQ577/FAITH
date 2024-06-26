if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=MCTEM

root_path_name=./dataset/electricity
data_path_name=electricity.csv
model_id_name=electricity
data_name=electricity



for pred_len in 96 192 336 720 
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 1 \
      --n_heads 4 \
      --d_model 32 \
      --d_ff 32 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --des 'Exp' \
      --train_epochs 30 \
      --patience 5\
      --itr 1 --batch_size 32 --learning_rate 0.1 \

done

