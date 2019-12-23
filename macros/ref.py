import pandas as pd
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.ngboost import NGBoost
from ngboost.scores import MLE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# 获取高能对撞粒子数据集
train_data = pd.read_csv("jet_simple_data/simple_train_R04_jet.csv")
# 获取特征列
features = train_data[['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz',
                       'jet_energy', 'jet_mass']]
# 随机切分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    features.values,
    train_data.label.values, test_size=0.2)
ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE(), natural_gradient=True,
              verbose=False)
# 拟合
ngb.fit(X_train, Y_train)
# 预测
Y_preds = ngb.predict(X_test)
test_data = pd.read_csv("jet_simple_data/simple_test_R04_jet.csv")
features = test_data[['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz',
                      'jet_energy', 'jet_mass']]
Y_test_data = ngb.predict(features)
with open("submmission.csv", "") as f:
    f.write("id,label\n")
    for jet_id, label in zip(test_data['jet_id'], Y_test_data):
        f.write(jet_id + "," + label + "\n")
Y_dists = ngb.pred_dist(X_test)
# 检验均方误差 test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)
# 检验负对数似然test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test.flatten()).mean()
print('Test NLL', test_NLL)
