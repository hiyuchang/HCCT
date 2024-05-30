gpu="7"
seed="1 2 3 4 5"
epochs="50"
alpha="1"
num_users="10"
n_cluster="5"
goal="cifar_alpha"

for s in $seed
do
    python main_cluster.py --seed $s  --dataset cifar --model cnn4 --gpu $gpu --epochs 50 \
    --num_users 10 --find_opt_D --imb_type halfnormal
    --strategy hcct --alpha $alpha
done



