### Train
aios上pretrain路径：/data/team/wjj/47_act_pythia_pire
aios上finetune路径: /data/private/wenjj/47_act_pythia_pire
1. 修改 ./scripts/train_act.sh 中的参数
    ~~~
   --load_pretrain 是否加载预训练的权重
   --pretrain_image_size 训练图片大小 320 or 480 
   --per_device_train_batch_size 30 default 32
   ~~~
2. 修改 train_act_pythia.py 中的参数(非必需)
3. 权重转换and下载权重(./scripts/download_weights.sh)

### Validation
#### 1.copy weights to WJJ1T 
WJJ1T_path="/media/eai/WJJ1T/droid/results/pythia_410M/vanilla_pythia_pt_f_vit_act_new_view"
#### 2.edit the hyper parameters in eval_act_vlm_real_franka.py
1. change the 'model_path' in policy_config 
2. change im_size(optional)
~~~
python eval_act_vlm_real_franka.py
~~~# moevla
