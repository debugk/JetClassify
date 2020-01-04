# 高能对撞粒子分类挑战赛
### [官网](https://www.biendata.com/competition/jet/)
### 简单赛道
已经截止。

利用jet 的px,py,pz,energy,mass,number_of_particle 区分不同的jet: d, c, b, gluon。

关键点: 同一个event ID 下的jet 拥有相同的label。 原因不明，可能跟产生机制有关。

计算同一个event 下jet 的不变质量, 并未发现明显共振峰，，，emmm, 估计产生道是 pp->pp(gg), gg->gg。

### 复杂赛道
截止日期2020.02.28

多了个jet中particle的信息，还没开始看。

## 运行环境全家桶
可以使用docker配置所需机器学习环境，主要用到numpy, pandas, keras。配置文件放在下面的git：
https://gitlab.cern.ch/fuhe/basic-python-image

linux 环境下推荐使用singularity:
```
singularity run -e docker://gitlab-registry.cern.ch/fuhe/basic-python-image:latest
```

其他环境可以docker, 使用时需要映射该code所在的文件夹的绝对路径：
```
docker pull gitlab-registry.cern.ch/fuhe/basic-python-image:latest # 下载镜像
docker run --rm -it -v ${path}:${path} gitlab-registry.cern.ch/fuhe/basic-python-image # 运行
```

## 画输入变量图
```
python3 ../JetClassify/macros/plotInput.py $inp --do-event -o plots_event
python3 ../JetClassify/macros/plotInput.py $inp -o plots
```

## 增加几个变量
也可以在训练的时候加，提前加后期调整训练参数的时候可以方便点：
```
python3 ../macros/plotInput.py simple_test_R04_jet.csv --do-add -o Input/ --outname="new_test_jet.csv"
python3 ../macros/plotInput.py simple_train_R04_jet.csv --do-add -o Input/ --outname="new_train_jet.csv"
```

## 训练
```
python3 ../JetClassify/macros/trainJetClassify.py ${train_file} -t ${test_file} -o RNN_xaxis_theta_p3_evt_v2 &> log_xaxis_theta_p3_evt_v2 & 
```
