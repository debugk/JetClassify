#train_file="Input/new_train_jet.csv"
#train_file="../macros/jet_simple_data/simple_train_R04_jet.csv"
#test_file="Input/test/new_train_jet.csv"
#test_file="../macros/jet_simple_data/simple_test_R04_jet.csv"

train_file="Input_xaxis/new_train_jet.csv"
test_file="Input_xaxis/new_test_jet.csv"


python3 ../macros/trainJetClassify.py ${train_file} -t ${test_file} -o DNN_xaxis_theta_p3 &> log_xaxis_theta_p3 & 
