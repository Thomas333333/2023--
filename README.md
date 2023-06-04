- [机器问答和信息检索大作业](#机器问答和信息检索大作业)
  - [信息检索部分](#信息检索部分)
    - [要求](#要求)
    - [前端优化](#前端优化)
    - [MetaSearch](#metasearch)
    - [检索排序算法](#检索排序算法)
      - [~~GPT回答结果~~（未实现）](#gpt回答结果未实现)
  - [信息检索部分](#信息检索部分-1)


# 机器问答和信息检索大作业

此项目为第四小组的机器问答和信息检索代码部分

## 信息检索部分

在浏览器中输入url：http://localhost:8080/contentse

### 要求
> ElasticSearch-7.6.1

### 前端优化

+ 类百度效果，通过监听表单输入来改变html元素比例

  ![20230604112841](https://cdn.jsdelivr.net/gh/Thomas333333/MyPostImage/Images/20230604112841.png)

  <center>图1-输入前</center>

  ![20230604112916](https://cdn.jsdelivr.net/gh/Thomas333333/MyPostImage/Images/20230604112916.png)

  <center>图2-输入后</center>
+ 搜索框输入一键清除

  ![20230604113437](https://cdn.jsdelivr.net/gh/Thomas333333/MyPostImage/Images/20230604113437.png)

  <center>图3-有输入时</center>

  ![20230604113446](https://cdn.jsdelivr.net/gh/Thomas333333/MyPostImage/Images/20230604113446.png)

  <center>图4-清除后</center>

+ 前端界面美化
  + Logo绘制：用inkscape绘制仿Google logo的彩色字母矢量图
  + 搜索按钮美化：
  + 返回搜索结果的列表美化：
  + 单条搜索结果的界面美化：


![20230604113604](https://cdn.jsdelivr.net/gh/Thomas333333/MyPostImage/Images/20230604113604.png)

图-5 搜索页面

![20230604113612](https://cdn.jsdelivr.net/gh/Thomas333333/MyPostImage/Images/20230604113612.png)

图-6 答案页面


### MetaSearch

+ 实时爬取Bing的搜索结果，并载入到ES数据库中

### 检索排序算法

+ 使用BM25排序算法

#### ~~GPT回答结果~~（未实现）

## 信息检索部分


