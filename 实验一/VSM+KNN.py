import os,sys
import math
from textblob import TextBlob
from textblob import Word
from nltk.corpus import stopwords

reload(sys) # Python2.5 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入 
sys.setdefaultencoding('utf-8') 

#预处理，划分数据集
def preprocess(path):
    files=os.listdir(path)
    a=0
    list=[]
    label=[]
    for i in files:
        zfiles=os.listdir(path+ os.path.sep +i)
        for zfile in zfiles:
            data=open(path + os.path.sep + i + os.path.sep + zfile)
            data=data.readlines()
            data=int(str(data).lower())#统一大小写
            words=data.words#分词
            print(words)
            filteredwords=[w for w in words if(w not in stopwords.words('english'))]#去停用词
            words=filteredwords.lemmatize()
            #建字典
            dic={}
            for i in words:
                if dic.get(i):
                    dic[i]+=1
                else:
                    dic[i]=1
            for k,v in dic.items():
                    dic[k]=(1+math.log(v))#标准化tf
                    #print(k,dic[k])
            list.append(dic)
            label[a]=zfile
            a+=1
    return list,label

def Vsm(prelist1,prelist2): #用向量表示文件数据
    for dic in prelist1:
        dic2 = {}
        m = 0
        for k,v in dic.items():
            if dic[k] >= 50:#保留词频较大的数据
                break
            m = m + 1
            dic2[k] = v
        list2.append(dic2)
    for dic in prelist2:
        dic3 = {}
        m = 0
        for k,v in dic.items():
            if dic[k] >= 50:#保留词频较大的数据
                break
            m = m + 1
            dic3[k] = v
        list3.append(dic3)
    for dic2 in list2:
        i1=0
        for k1,v1 in dic2.items():
            j1=0
            traindata[i1][j1]=v1
            for dic3 in list2:
                i2=0
                for k2,v2 in dic3.items():
                    j2=0
                    if dic3[k1]:
                        test[i2][j2]=v2
                    else:
                        test[i2][j2]=0
                    j2+=1
                i2+=1
            j1+=1
        i1+=1
    return train,test
        
def KNN(traindata,trainlabel,testdata,testlabel):#使用KNN分类
    Accuracy=0
    k=5
    #求余弦 = AB/|A|*|B|
    for i in traindata:  #求训练集数据的模
        mosum = 0
        for key in i[1]:
            mosum = mosum + i[1][key] * i[1][key]
        mo = math.sqrt(mosum)
        list.append(mo)
    motrain=list
    for i in testdata:  #求测试集数据的模
        mosum = 0
        for key in i[1]:
            mosum = mosum + i[1][key] * i[1][key]
        mo = math.sqrt(mosum)
        list.append(mo)
    motest=list
    for i in range(len(testdata)):
        cosi = []     
        cosclass = [] 
        for j in range(len(traindata)):
            sum = 0 
            for key in testdata[i][1]:
                if key in traindata[j][1]:
                    sum = sum + testdata[i][1][key] * traindata[j][1][key]
            
            cosij = sum /( motrain[j] * motest[i])
            
        #求出前k个数据
        if len(cosi) < k:
            cosi.append(cosij)
            cosclass.append(trainlabel[j])
        elif cosij > min(cosi):
            t = cosi.index(min(cosi))
            cosi[t] = cosij
            cosclass[t] = trainlabel[j]
        #找k个数据中出现次数最多的类   
        class = ' '
            max = 0
            cou = collections.Counter(cosclass) #counter函数
            for k,v in cou.items():
                if v > max:
                    class = k
                    max = v
        if class == testlabel[i]:
            Accuracy+=1
    
    #求准确率
    rightnum=Accuracy
    Accuracy=Accuracy/len(testlabel)
    print('测试结果：')
    print('总测试次数：' , len(testlabel))
    print('预测成功次数：' , rightnum)
    print('预测准确率：' , Accuracy)

if __name__ == "__main__":
    path="data\traindata"
    [prelist1,trainlabel]=preprocess(path)
    path="data\testdata"
    [prelist2,testlabel]=preprocess(path)
    [traindata，testdata]=Vsm(prelist1,prelist2)
    KNN(traindata,trainlabel,testdata,testlabel)
    