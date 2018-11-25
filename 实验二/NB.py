# -*- coding: utf-8 -*-
import os
import collections
import math

trainset_path='D:\\a\\data mining\\traindata' #训练集数据路径
testset_path='D:\\a\\data mining\\testdata' #测试集数据路径

#统计训练集内的文档数目（计算每个类别出现概率时用得到）
def count_files(trainset_path):
    count=0
    floderlist=os.listdir(trainset_path)
    for floder in floderlist:
        floderpath=trainset_path+os.path.sep+floder
        filelist=os.listdir(floderpath)
        count=count+len(filelist)
    return count


#为训练集创建向量：每一个类别为一个向量，向量的结构为[类名，类的词典长度（即类中所有单词的总数），该类别出现的概率，该类的字典] 即[string,int,float,{}]
def get_trainVectors():

    totalfiles = count_files(trainset_path)
    floderlist = os.listdir(trainset_path)

    vectorlists =[] #存放整个训练集所有向量的列表

    j = 0 #为了显示向量个数
    for floder in floderlist:

        j = j+1

        vector=[] #一个向量
        vector.append(floder) #追加类名
        totalwords = 0 #为每个类的所有单词计数
        filecount = 0.0 #统计该类中的文档数目
        wordsdict = {} #该类的词典（包含该类的所有单词,value为该单词在类中出现的总次数）

        floderpath=trainset_path+os.path.sep+floder
        filelist=os.listdir(floderpath)
        for file in filelist:
            filecount += 1
            filepath = floderpath+os.path.sep+file
            lines = open(filepath).readlines() #len(lines)即为该文档中的单词总数
            totalwords += len(lines)
            wordcountdic = collections.Counter(lines) #词典中为文档中的所有词，以及在该文档中出现的总次数
            for key, value in wordcountdic.items():
                key = key.strip()
                wordsdict[key] = wordsdict.get(key, 0)+value
        vector.append(totalwords)
        p = filecount / totalfiles
        vector.append(p)
        vector.append(wordsdict)
        # print('第'+str(j)+'个向量：')
        # print(vector)
        vectorlists.append(vector)
    return vectorlists


#将测试集的每一个文档表示成向量，[类名，以该文档所有单词为元素的列表]。
def get_testVectors():

    trainVectors=[] #存放测试集所有向量的列表
    i=0 #统计向量个数

    floderlist=os.listdir(testset_path)
    for floder in floderlist:
        floderpath = testset_path + os.path.sep +floder
        filelist=os.listdir(floderpath)
        for file in filelist:

            i+=1

            vector=[] #一个文档表示成一个向量
            vector.append(floder) #追加类名
            words=[] #存放一个文档内所有的单词

            filepath=floderpath + os.path.sep +file
            lines=open(filepath).readlines()
            wordcountdic=collections.Counter(lines)
            for key,value in wordcountdic.items():
                key=key.strip()
                words.append(key)
            vector.append(words)
            # print('第'+str(i)+'个向量：')
            # print(vector)
            trainVectors.append(vector)
    return trainVectors

def get_allwords():
    allwords=0
    floderlist = os.listdir(trainset_path)
    for floder in floderlist:
        floderpath = trainset_path + os.path.sep + floder
        filelist = os.listdir(floderpath)
        for file in filelist:
            filepath = floderpath + os.path.sep + file
            lines = open(filepath).readlines()  # len(lines)即为该文档中的单词总数
            allwords += len(lines)
    return allwords


def NB():
    trainVectorList=get_trainVectors()
    testVectorsList=get_testVectors()

    allwords=get_allwords() #allwords为数据集内单词总数
    success = 0 #统计成功次数
    count=0 #记录测试集文档总数

    for i in range(len(testVectorsList)): #待分类的每一个文档
        count+=1
        eachclassp = []  # 存放一个文档在每个类别中的后验概率

        for j in range(len(trainVectorList)): #每一个类别
            p = trainVectorList[j][2] #概率初值设为该类别出现的概率
            p = math.log10(p)

            for word in testVectorsList[i][1]:
                numerator = float(trainVectorList[j][3].get(word, 0)+1)
                denominator = float(trainVectorList[j][1]+allwords)
                wordp = math.log10(numerator/denominator)
                p += wordp #算出了一个文档属于一个类别的概率

        judgeclass = eachclassp[0][0]
        if judgeclass == testVectorsList[i][0]:
            success = success+1
    print('测试结果：')
    print ('总测试次数：' + str(len(testVectorsList)))
    print ('预测成功次数：'+str(success))
    successrate = float(success)/len(testVectorsList)
    print ('预测准确率：'+str(successrate))

NB()





