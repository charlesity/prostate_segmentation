
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data_roc = np.load('./Results/Max_Entropy_Pool_ROC_'+ 'Experiment_' + str(0) + '.npy')
data_fpr = np.load('./Results/Max_Entropy_Pool_FPR'+ 'Experiment_' + str(0) + '.npy')
data_tpr = np.load('./Results/Max_Entropy_Pool_TPR'+ 'Experiment_' + str(0) + '.npy')


for i in range(len(data_fpr)):
    plt.plot(data_fpr[i], data_tpr[i], label='ROC curve (area = %0.2f at interaction = %1.0f)' % (data_roc[i], i))
plt.plot([0, 1], [0, 1], color='navy', lw = 2,  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - ROC Curve')
plt.legend(loc="lower right")
plt.show()

