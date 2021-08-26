export CUDA_VISIBLE_DEVICES=2

python main.py --batch_size=64 \
    --cr_loss_weight=5.08720932695335 --cutmix_prob=0.7615524094697519 --cutmix_warmup_iter=1000 \
    --d_lr=7.122979672016055e-05 --g_lr=0.00018083340390609657 --grad_gen_loss_weight=0.11960717521104237 \
    --grad_loss_weight=35.310016043755894 --img_gen_loss_weight=0.14178356036938378 --max_iter=100000 \
    --model_name=DUGAN --num_channels=32 --num_layers=10 --num_workers=32 --pix_loss_weight=5.034293425614828 \
    --run_name=official --save_freq=2500 --test_batch_size=1 --test_dataset_name=cmayo_test_512 \
    --train_dataset_name=cmayo_train_64 --use_grad_discriminator=true --weight_decay 0. --num_workers 4
