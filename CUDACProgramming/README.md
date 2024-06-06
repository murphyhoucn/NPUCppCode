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
    - 谭升的博客: https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89
    - CUDA C Programming code: https://github.com/kriegalex/wrox-pro-cuda-c
    
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

- CUDA编程模型中两个主要的抽象概念：内存层次结构和线程层次结构。

- CUDA采用单指令多线程（SIMT）架构来管理和执行线程，每32个线程为一组，被称为线程束（warp）。线程束是SM中基本的执行单元。

- 在同一个线程束中的线程执行不同的指令，被称为线程束分化。线程束分化会导致性能明显地下降。

- 带宽与吞吐量，都是用来度量性能的指标。
    - 带宽通常是理论峰值。
    - 吞吐量是指已达到的值。
    - 带宽通常是用来描述单位时间内最大可能的数据传输量。
    - 吞吐量是用来描述单位时间内任何形式的信息或操作的执行速度，例如，每个周期内完成多少个指令。

- 目前主流的 CUDA 驱动不再支持nvprof命令!
``` bash
(base) houjinliang@3080server:~/MyProject/NPUCppCode/CUDACProgramming$ nvprof ./main 
======== Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
                  Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
                  Refer https://developer.nvidia.com/tools-overview for more details.
```
    - 目前主流的 CUDA 驱动不再支持nvprof命令，但我们仍可以在 NVIDIA Nsight Systems 中使用，在终端输入 `nsys nvprof ./*.o`就可以看到CUDA 程序执行的具体内容。
    - 另外，`nvprof --metrics` 命令的功能被转换到了 `ncu --metrics` 命令中，下面就对 nvprof/ncu --metrics命令的参数作详细解释，nsys 和 ncu 工具都有可视化版本，这里只讨论命令行版本。
    - https://zhuanlan.zhihu.com/p/666242337
    - https://www.cnblogs.com/peihuang/p/17665525.html

- CUDA编程是与硬件紧密相关的，我所有的GPU是RTX 3080
    - GA102 白皮书: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
    - RTX 3080 规格说明：https://www.bilibili.com/video/BV1oV41127q5/?spm_id_from=333.337.search-card.all.click&vd_source=6d46640a443a49f050af078d1f65143e
    - RTX 3080 GPU采用的是安培架构，核心代号GA102，但RTX 3080并没有用到完整的GA102,而是在完整的GA102的基础上阉割掉了一个GPC，所以可用SM之后68组。
    - 所使用的设备的信息可用下面两个程序查看：`cpt2_check_device_infor.cu`, `cpt3_simple_divice_query.cu`

- 网格与线程块的配置准则
    - 保持每个块中线程数量是线程束大小（32）的倍数
    - 避免块太小：每个块至少要有125或256个线程
    - 根据内核资源的需求调整块大小
    - 块的数量要远远多于SM的数量，从而在设备中可以显示有足够的并行
    - 通过实验得到最佳执行配置和资源使用情况

- 并行性的表现
    - 一个内核的可实现占用率被定义为：每周期内活跃线程束的平均数量与一个SM支持的线程束最大数量的比值。更高的占用率并不一定意味着有更高的性能。
    - 更高的加载吞吐量并不意味之更高的性能。
    - 一个块的最内层维数（block.x）应该是线程束大小的倍数。线程块最内层维度的大小对性能起着关键的作用！

- CUDA编程的性能指标
    - 在大部分情况下，一个单独的指标不能产生最佳的性能。
    - 与总体性能最直接相关的指标或事件取决于内核代码的本质。
    - 在相关的指标与事件之间寻求一个好的平衡。
    - 从不同角度查看内核以寻找相关指标之间的平衡。
    - 网格/块启发式算法为性能调节提供了一个很好的起点。

- 避免分支分化
    - 在向量中执行满足**交换律和结合律**的运算，被称为**规约问题**。
    - 并行规约问题。
    