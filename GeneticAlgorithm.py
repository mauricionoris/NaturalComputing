
# coding: utf-8

# In[21]:


data = {
    
  'model'     :  None
, 'population':  None
, 'truthtable':  None
, 'fitness'   :  None
, 'mutation'  :  None   

}

params = {
     'N'         : 64 # numero de indivíduos
,    'Z'         : 500 # numero de gerações
,    'p_fertil'  : 0.5 # individuos ferteis da geracao
,    'p_mut'     : 0.1 # prob mutação
,    'p_fitness' : 0.3 #prob fitness    
,    'shape'     : (28,28) # tamanho da figura
,    'index'     : 20 # id modelo
}


# In[3]:


#Preparação

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

from numpy.linalg import norm

import tensorflow as tf
mnist = tf.keras.datasets.mnist


# In[4]:


def setPopulation():
    data['population'] = np.full((params['n_genes'], params['N'], params['Z']+1), np.inf)
    data['truthtable'] = np.full((params['n_genes'], params['N'], params['Z']+1), np.inf)
    data['fitness']    = np.full((params['n_genes'], params['N'], params['Z']), np.inf)
    data['mutation']   = np.full((params['n_genes'], params['N'], params['Z']), np.inf)
        


# In[26]:


def setModel(): 
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    
    data['model'] = np.array(x_train[params['index']], dtype='float').flatten()
    data['model'][data['model']>0] = 1 # normalizacao (somente preto e branco)
    
    params['n_genes'] = data['model'].shape[0]
        


# In[6]:


def showImage(input):
    plt.imshow(input.reshape(params['shape']), cmap='gray')
    plt.show()


# In[7]:


def showImage2(gen):

    msg = "Geração: {} - Indivíduo: {} - Best Fitness: {}"

    ix = np.where(np.isin(data['fitness'][0,:,gen], max(data['fitness'][0,:,gen])))[0]
    print (msg.format(gen,ix[0],data['fitness'][0,ix[0],gen]))
    
    x1 = np.arange(params['N'])
    y1 = data['fitness'][0,:,gen]
   
    plt.plot(x1, y1, 'o-')
    plt.title('Fitness da geração:' + str(gen))
    plt.ylabel('%')
    plt.show()

    plt.imshow(data['population'][:,ix[0],gen].reshape(params['shape']), cmap='gray')
    plt.xlabel('Individuo')
    plt.show()


# In[8]:


def initGeneration(gen):
    data['population'][:,:,gen] = np.random.choice([0,1],data['population'][:,:,gen].shape)


# In[9]:


def calcFitness(ind,gen):
    data['truthtable'][:,ind,gen] = data['model'] == data['population'][:,ind,gen]
    data['fitness'][0,ind,gen] = np.sum(data['truthtable'][:,ind,gen],axis=0)/params['n_genes']
    


# In[10]:


def selFerteis(gen):
    array = -np.array(data['fitness'][0,:,gen])
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    data['fitness'][1,:,gen] = ranks < params['p_fertil']*params['N']


# In[11]:


def crossover(gen):

    ferteis = np.where(np.isin(data['fitness'][1,:,gen],[True]))
    
    temp = []
    temp2 = []
    temp3 = []
    
    for idf in ferteis[0]:
        temp.append(data['population'][:,idf ,gen]);
        temp2.append(np.where(np.isin(data['truthtable'][:,idf, gen],[False] ))[0]);
    
    for idx in range(0,len(temp),2):
        f1 = temp[idx]
        f2 = temp[idx+1]
        bdg_f1 = temp2[idx]
        bdg_f2 = temp2[idx+1]
        
        point = np.random.randint (params['n_genes'])
        #point = int(params['n_genes']/2)
        
        f3 = np.concatenate((f1[:point], f2[point:]))
        f4 = np.concatenate((f2[:point], f1[point:]))   

        new_f1 = f1
        for i in bdg_f1:
            new_f1[i] = np.random.choice([0,1])
            #new_f1[i] = f2[i]

        new_f2 = f2
        for i in bdg_f2:
            #new_f2[i] = np.random.choice([0,1])
            new_f2[i] = f1[i]  
        
        temp3.append(new_f1)
        temp3.append(new_f2)
        temp3.append(f3)
        temp3.append(f4)
        
    for idx in range(0,len(temp3)):
        novoIndividuo(temp3[idx], idx, gen+1)
    


# In[12]:


def novoIndividuo(individuo, n, z):
    #print(individuo)
    data['population'][:, n, z] = mutation(individuo,n,z)
    


# In[13]:


def mutation(individuo,n,z):
    if np.random.choice([0, 1], p=[1-params['p_mut'],params['p_mut']]):
        gM = np.random.randint(params['n_genes'])
        data['mutation'][gM,n,z] = True
        individuo[gM] = 1 - individuo[gM]
        
    return individuo
    


# In[14]:


def evolution():
    
    for i in range(0, params['Z']-1):
        for j in range(0, params['N']):
            calcFitness(j,i)
        selFerteis(i)
        crossover(i)  

        if i % 100 == 0: 
            showImage2(i)


# In[27]:



setModel();
showImage(data['model']) 
setPopulation();
initGeneration(0);
evolution()


# In[23]:


data['model']


# In[25]:


data['population']

