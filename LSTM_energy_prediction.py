import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#获取训练集
def get_train_data(batch_size,time_step,data_train):
    batch_index=[]
    # data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    train_x,train_y=[],[]
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:16]
       # print(x)
       y=normalized_train_data[i:i+time_step,16,np.newaxis]
       # print(y)
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#获取测试集
def get_test_data(time_step,data_test):    
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:16]
       y=normalized_test_data[i*time_step:(i+1)*time_step,16]
       test_x.append(x.tolist())
       test_y.extend(y)
    return mean,std,test_x,test_y



#——————————————————定义神经网络变量——————————————————
def lstm(X):
    
    #——————————————————定义神经网络变量——————————————————
    #输入层、输出层权重、偏置
    weights={
             'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
             'out':tf.Variable(tf.random_normal([rnn_unit,1]))
            }
    biases={
            'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
            'out':tf.Variable(tf.constant(0.1,shape=[1,]))
           }
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#——————————————————训练模型——————————————————
def train_lstm(batch_size,time_step,train_data_):#batch_size=60,time_step=15    
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_data_)
    pred,_=lstm(X)
    # loss = tf.sqrt(tf.losses.mean_squared_error(tf.reshape(pred,[-1]),tf.reshape(Y, [-1]))) # rmse
    loss = tf.losses.mean_squared_error(tf.reshape(pred,[-1]),tf.reshape(Y, [-1])) # mse
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)#保存最近的15个模型
    #module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        minn = 1000
        for i in range(201):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_,minn)
            if min(loss_,minn) == loss_:
                print("保存模型：",saver.save(sess,'model_file6' + os.sep+'/stock2.model',global_step=i))
            if min(loss_,minn) < 0.02:
                print("保存模型：",saver.save(sess,'model_file_1' + os.sep+'/stock2.model',global_step=i))
            minn = min(loss_,minn)


#————————————————预测模型————————————————————
def prediction(time_step,data_test):
# def prediction(time_step=36):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step,data_test)
    pred,_ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_file6' + os.sep)
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)):
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})
            predict=prob.reshape((-1))
            test_predict.extend(predict)
        test_y=np.array(test_y)*std[16]+mean[16]
        test_predict=np.abs(np.array(test_predict)*std[16]+mean[16])        
        test_day = []
        test_pre = []
        for i in range(len(test_y)//48):
            test_day.append(np.sum(test_y[i*48:(i+1)*48]))
        for i in range(len(test_y)//48):
            test_pre.append(np.sum(test_predict[i*48:(i+1)*48]))
        datafra = pd.DataFrame({'value':test_pre})
        #将DataFrame存储为csv,index表示是否显示行名，default=True
        datafra.to_csv(csv_name,index=False,sep=',')

        plt.plot(list(range(len(test_predict))),test_predict,label='prediction',linewidth=1.5,marker='o',markersize=5,color='r')
        plt.plot(list(range(len(test_y))),test_y,label='true',linewidth=1.5,marker='*',markersize=5,color='c')
        plt.xlabel('day') 
        plt.ylabel('value')
        plt.title('prediction of days') 
        plt.legend()
        plt.show()

        MAPE = []
        MAPE_ = []
        for i in range(len(test_y)):
            MAPE.append((test_y[i] - test_predict[i]) / test_y[i] * 100)
            MAPE_.append(np.abs(test_y[i] - test_predict[i]) / test_y[i] * 100)
        MAPE_meanday = np.mean(MAPE)
        MAPE_mean_ = np.mean(MAPE_)
        print("MAPE:",MAPE_meanday,"%")
        print("|MAPE|:",MAPE_mean_,"%")
        print("总的MAPE:",(np.sum(test_y) - np.sum(test_predict))/ np.sum(test_y),"%")
        print("sum_test",np.sum(test_y))
        print("sum_pred", np.sum(test_predict))
        from sklearn.metrics import r2_score
        print('R2_score:',r2_score(test_y,test_predict))

def train(train_data_,data_test):
    with tf.variable_scope('train'):
        train_lstm(10,7,train_data_)
    

    with tf.variable_scope('train',reuse=True):
        prediction(1,data_test)

if __name__ == '__main__':    
    df = pd.read_csv("./DayLoadSet.csv")
    data=df.iloc[:511,1:].values
    data_test=df.iloc[510:,1:].values
    global rnn_unit,input_size,output_size,lrcsv_name
    tf.reset_default_graph()
    rnn_unit= 12
    input_size=16
    output_size=1
    lr=0.0006
    csv_name='test_pre1.csv'
    train(data,data_test)
