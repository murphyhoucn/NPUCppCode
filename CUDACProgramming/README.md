# CUDA C Programming

## 参考资料
- 《CUDA C 编程权威指南》[美] 马克斯·格罗斯曼（Max Grossmen）& [美] 泰·麦克切尔（Ty McKercher） 机械工业出版社  2017年出版
    - 第1章　基于CUDA的异构并行计算
    - 第2章　CUDA编程模型
    - 第3章　CUDA执行模型
    - 第4章　全局内存
    - 第5章　共享内存和常量内存
    - 第6章　流和并发
    - 第7章　调整指令级原语
    - 第8章　GPU加速库和OpenACC
- 

## 项目环境
- Ubuntu 20.04 LTS (Host) / Ubuntu 18.04 LTS (Server) 
- Visual Studio Code, Remote-SSH, C/C++, Nsight Visual Studio Code Edition
- CUDA-11.3, cuDNN 8.9

## 参考链接
- 一个不错的博客网站：https://godweiyang.com/2021/01/25/cuda-reading/
    - CUDA编程入门极简教程: https://zhuanlan.zhihu.com/p/34587739 ✔
    - 《CUDA C Programming Guide》《CUDAC编程指南》导读：https://zhuanlan.zhihu.com/p/53773183
    - CUDA编程入门系列：https://zhuanlan.zhihu.com/p/97044592
    
- 代码风格采用Google开源项目风指南-C++风格指南：https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/contents.html

## 简单笔记

### 第1章　基于CUDA的异构并行计算
- 第一章概述了CUDA编程的一些基础知识，如串行编程和并行编程、CPU与GPU的异构计算、来自GPU的hello world!
- 第一章的习题没什么大问题
 - 程序结束时： 在程序即将结束时调用 cudaDeviceReset() 可以确保释放所有CUDA资源，以避免内存泄漏或其他资源泄漏。
 - 如果程序结束时没有调用 cudaDeviceReset()，程序就没有输出“hello world from gpu”了。这是为什么？❓️

### 第2章　CUDA编程模型
- 这一章的重点内容在*CUDA的两级线性层次结构——grid和block*。⭐️
- ⭐️组织线程是CUDA编程的重点之一，但看完了这一章感觉有点儿稀里糊涂的，没有搞懂到底是怎么组织线程，怎么找到最佳执行配置！
- 线程和块索引映射到矩阵坐标上，矩阵坐标映射到全局内存中的索引/存储单元上的两个公式，很重要！⭐️
``` c
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * knx + ix;
```
- 二维网格与二维块，二维网格与一维块，一维网格与二维块，一维网格与一维块
- 第二章的习题。。。额。。不太会做啊！😭

### 第3章　CUDA执行模型第2章　CUDA编程模型

