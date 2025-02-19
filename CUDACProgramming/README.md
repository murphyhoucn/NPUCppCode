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

### VSCode NVCC Configure
```bash
(base) houjinliang@3080server:~/MyDevProject/NPUCppCode/CUDACProgramming/.vscode$ ll
总用量 24
drwxrwxr-x 2 houjinliang houjinliang 4096 5月   5  2024 ./
drwxrwxr-x 4 houjinliang houjinliang 4096 2月  19 15:47 ../
-rw-rw-r-- 1 houjinliang houjinliang  453 5月   4  2024 c_cpp_properties.json
-rw-rw-r-- 1 houjinliang houjinliang  267 5月   4  2024 launch.json
-rw-rw-r-- 1 houjinliang houjinliang  541 5月  15  2024 settings.json
-rw-rw-r-- 1 houjinliang houjinliang  228 5月   4  2024 tasks.json
```
配置好了之后，可以在VSC下直接对“.cu”文件编译并进入DeBUG。

> 在Visual Studio Code（VSCode）编辑器中，`.vscode`目录用于存储特定于项目的配置文件。以下是对提到的四个文件的解释：

`c_cpp_properties.json`:这个文件是C/C++扩展的一部分，用于配置C/C++项目的 IntelliSense。IntelliSense 是VSCode中用于代码补全、参数信息、快速信息和成员列表的功能。它包含了诸如包含路径、定义、编译器路径和 IntelliSense 模式等配置，这些配置有助于VSCode正确解析和索引C/C++代码。

`launch.json`:这个文件用于配置VSCode的调试器。它定义了调试会话的属性，例如调试器类型（例如 GDB、LLDB、Python、Node.js 等）、程序启动命令、工作目录、环境变量、调试设置和断点等。用户可以通过修改这个文件来定制调试行为，例如设置程序启动参数、选择调试器、指定要调试的程序等。

`settings.json`:这个文件包含了项目特定的VSCode设置。在这里设置的配置会覆盖用户和工作区级别的设置。它可以包含各种各样的设置，比如编辑器行为、代码格式化选项、插件配置、文件关联等。

`tasks.json`:这个文件用于配置和管理VSCode中的任务。任务可以是任何外部工具，如编译器、构建脚本、命令行工具等。在这个文件中，你可以定义任务、它们的命令、参数、问题匹配器（用于捕获输出中的错误和警告）以及任务执行时的其他选项。

这些文件共同为C/C++项目提供了一个完整的开发环境配置，允许开发者自定义编辑、调试和构建过程。通过在.vscode目录下配置这些文件，开发者可以在不同的机器或不同的开发环境中轻松重现相同的开发体验。


若是不借助VScode的配置，直接使用命令行的nvcc直接对“.cu”进行编译
``` bash
# 不进入调试（下面两个命令一样，仅仅是参数顺序不同）
nvcc cpt1_hello_from_gpu.cu -o main 
nvcc -o main cpt1_hello_from_gpu.cu
./main 

# 进入调试
nvcc -g -G -o main  cpt1_hello_from_gpu.cu
gdb ./main 
```



## 参考链接
- 一个不错的博客网站：https://godweiyang.com/2021/01/25/cuda-reading/
    - CUDA编程入门极简教程: https://zhuanlan.zhihu.com/p/34587739 ✔
    - 《CUDA C Programming Guide》《CUDA C 编程指南》导读：https://zhuanlan.zhihu.com/p/53773183
    - CUDA编程入门系列：https://zhuanlan.zhihu.com/p/97044592
    - 谭升的博客: https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89
    - CUDA C Programming code: https://github.com/kriegalex/wrox-pro-cuda-c
    
- 代码风格采用Google开源项目风指南-C++风格指南：https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/contents.html

## CUDA 基础 - 谭升的博客

### Kernel核函数编写有以下限制
- 只能访问设备内存
- 必须有void返回类型
- 不支持可变数量的参数
- 不支持静态变量
- 显示异步行为




## CUDA_C_编程权威指南

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
    