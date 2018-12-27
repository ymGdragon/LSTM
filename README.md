# LSTM
数据集解释下：


F_day1：前一天value值 \
F_week: 前一周的当天的value值（今天周一，前一周周一的值）\
dayofweek: 周几\
isWorkday: 是否工作日\
isHoliday: 是否节假日\
Tem_max: 温度最高值\
Tem_min: 温度最低值\
RH_max: 湿度最高值\
RH_min: 湿度最低值\
（这里4个用最值是因为给出的是1个小时1个值，按照天计算的话，选了最高和最小值，其他值比较小）\
Tag: \
kmeans: 温度湿度四个值的聚类分类标签\
Value: 用电量

Value是需要预测的，其他为特征列
划分了7:3的比例进行预测。
结果为：
