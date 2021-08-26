export CUDA_VISIBLE_DEVICES=2

python main.py --batch_size 256 \
    --max_iter 20000 --save_freq 2000 --train_dataset_name cmayo_train_64 --test_dataset_name cmayo_test_512 \
    --hu_min -300 --hu_max 300 --weight_decay 0 --test_batch_size 16 --model_name REDCNN \
    --run_name official --num_layers 10 --num_channels 32 --init_lr 1e-4 --min_lr 1e-5