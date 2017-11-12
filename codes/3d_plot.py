import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
x = range(10,200,10)

#[NN1,NN2,NN3,SVM_linear,SVM_gauss,SVM_sigmoid,SVM_poly5,SVM_poly3,SVM_poly2,SVM_poly1,logistic_reg]
ylim=(0.4, 1.0)

NN1_line = mlines.Line2D([], [], color='#97b21e', marker='o',
                          markersize=5, label='NN[60,6]')
NN2_line = mlines.Line2D([], [], color='#228d9b', marker='*',
                          markersize=5, label='NN[600,60,6]')
NN3_line = mlines.Line2D([], [], color='#1e4c96', marker='.',
                          markersize=5, label='NN[1500,100,10]')
SVM_linear = mlines.Line2D([], [], color='#9b4c1f', marker='^',
                          markersize=5, label='SVM Linear')

SVM_gauss = mlines.Line2D([], [], color='#287754', marker='>',
                          markersize=5, label='SVM gauss')

SVM_sigmoid = mlines.Line2D([], [], color='#48125b', marker='1',
                          markersize=5, label='SVM sigmoid')

SVM_poly5 = mlines.Line2D([], [], color='#3f0f25', marker='2',
                          markersize=5, label='SVM poly5')

SVM_poly3 = mlines.Line2D([], [], color='#7c7c62', marker='3',
                          markersize=5, label='SVM poly3')

SVM_poly2 = mlines.Line2D([], [], color='#384942', marker='4',
                          markersize=5, label='SVM poly2')
SVM_poly1 = mlines.Line2D([], [], color='#183238', marker='8',
                          markersize=5, label='SVM poly1')
logistic_reg = mlines.Line2D([], [], color='#242447', marker='p',
                          markersize=5, label='SVM poly1')

plt.figure(figsize=(15,15))
plt.ylim(*ylim)
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.xticks(x)
plt.yticks(np.linspace(.4,1,50))
plt.legend(handles=[NN2_line,NN3_line,SVM_linear,SVM_gauss,SVM_sigmoid,SVM_poly5,SVM_poly3,SVM_poly2,SVM_poly1,logistic_reg])
plt.plot(x,classifier_data[1],marker="*",color='#228d9b')
plt.plot(x,classifier_data[2],marker=".",color='#1e4c96')
plt.plot(x,classifier_data[3],marker="^",color='#9b4c1f')
plt.plot(x,classifier_data[4],marker="<",color='#287754')
plt.plot(x,classifier_data[5],marker="1",color='#48125b')
plt.plot(x,classifier_data[6],marker="2",color='#3f0f25')
plt.plot(x,classifier_data[7],marker="3",color='#7c7c62')
plt.plot(x,classifier_data[8],marker="4",color='#384942')
plt.plot(x,classifier_data[9],marker="8",color='#183238')
plt.plot(x,classifier_data[10],marker="p",color='#242447')
plt.grid(True)
plt.savefig("figures1_new", dpi = 400)
