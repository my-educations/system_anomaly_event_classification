## [Deep Convolutional Neural Networks for Anomaly Event Classification on Distributed Systems](https://arxiv.org/pdf/1710.09052v1.pdf)


### * Author: Jiechao Cheng, Rui Ren, Lei Wang and Jianfeng Zhan


To assist in focusing on the classification and the prediction of anomaly events, and gaining critical insights from system  event  records,  we  propose  a  novel  log  preprocessing method  which  is  very  effective  to  filter  abundant  information and  retain  critical  characteristics.  Additionally,  a  competitive approach  for  automated  classification  of  anomalous  events
detected  from  the  distributed  system  logs  with  the  state-of-the-art  deep  (Convolutional  Neural  Network)  architectures  is proposed  in  this  paper.  We  measure  a  series  of  deep  CNN algorithms with varied hyper-parameter combinations by using
standard evaluation metrics, the results of our study reveals the advantages and potential capabilities of the proposed deep CNN
models  for  anomaly  event  classification  tasks  on  real-world systems.  The  optimal  classification  precision  of  our  approach is  98.14%,  which  surpasses  the  popular  traditional  machine learning  methods.



![figure](https://github.com/stephen-cheng/system_anomaly_event_classification/blob/master/code/method_cnn/figure/figure6.png)
            Deep CNN models with different hyper-parameter combinations show diverse recognition results.

### dataset:

https://github.com/stephen-cheng/system_anomaly_event_classification/tree/master/code/log2msg


### convert event text to numeric feature:

https://github.com/stephen-cheng/system_anomaly_event_classification/tree/master/code/msg_classification


### adaboost, decision tree, mlp, naive bayes, random forest, svm, etc. methods for anomaly event classification:

https://github.com/stephen-cheng/system_anomaly_event_classification/tree/master/code/method_ml

### diversified hyper-parameter combinations of CNN approach for anoamly event classification:

https://github.com/stephen-cheng/system_anomaly_event_classification/tree/master/code/method_cnn
