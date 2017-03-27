# xgboost源码结构

## include/
### xgboost/
* **base.h**
* **data.h**
* **feature_map.h**
* **gbm.h**
* **learner.h**
* **logging.h**
* **metric.h**
* **objective.h**
* **tree_model.h**
* **tree_updater.h**


## src/
* **cli_main.cc**
  xgboost程序的命令行接口程序,不包括在动态链接库中.
* **learner.cc**
  学习算法的实现.包括DumpModel, 对特定目标函数进行梯度提升的学习器LearnerImpl,
  在训练和预测过程中均有使用.InitModel, UpdateOneIter等.
* **learner.cc**
  日志器的实现,包括ConsoleLoggoer, TrackerLogger.
  

### common/
* **base64.h**
   base64 的输入/输出数据流支持，base64易于以文本格式在mapreduce中存储和传递
 * **bitmap.h**
   由点矩阵组成的数字图像
 * **common.h**
   通用工具,包括根据分隔符分割字符串,将任何数据转换成字符串
 * **config.h**
   由文件导入配置(configures)的帮助类,包括ConfigReaderBase, ConfigureStreamReader
 * **group_data.h**
   定义了按整型键值将数据分组的工具. 用来从无序输入中构建CSR/CSC矩阵
   输入: (key, value),(k1, v1),(k2, v2)
   输出: 数据数组[v1, v2 ... vn]; 分组指针.其中data[ptr[k]:ptr[k+1]]包含对应于键k的所有值.
 * **io.h**
   I/O序列化的通用流接口.包括支持额外PeakRead操作的输入流类
 *  **math.h**
   额外的数学工具. 包括各种内联函数Sigmoid ,就地Softmax, FindMaxIndex, LogSum, 按降
   序对pairs进行排序的比较函数CmpFisrt, ComSecond, CheckNAN, LogGamma.
 * **quantile.h**
   计算分位数的工具.
 * **random.h**
   随机相关的工具,使用std::mt19337梅森旋转演算法作为默认的随机数引擎(种子发生器).
 * **sync.h**
   重定向到通信框架*rabit*头文件的同步模块.
   Rabit简化了MPI的设计,抽取出机器学习最需要的Allreduce和Broadcast操作并加入了容灾
   的支持,使得基于分布式的同步通信(BSP)的机器学习算法可以在部分节点出错或丢失的情况
   下快速恢复计算.完成剩下的任务.
 * **thread_local.h**
   线程本地存储TLS的通用工具. 存储线程本地变量, 返回一个类型为T的线程本地单件(singleton).
   TLS的作用能将数据和特定的线程联系起来,可以保证当我们在多线程程序中访问全局变量或静态
   变量时不互相影响.
 * **common.cc**
   启用在common 命名空间中的各种全局变量.
### data/
* **data.cc**
  xgboost/data.h中的定义
* **simple_csr_source.h**
  数据源的最简单形式,可以用来创建DMatrix,继承自*DataSource*.
  这是一种按行存储的内存(in-memory)数据结构.CSR(Compressed Sparse Row)按行压缩存
  储的稀疏矩阵的一种存储格式,一个矩阵M由三个一维数组组成, 分别代表非零数值(nonzero -
  values),行偏移(the extends of row),列号(column indice).
* **sparse_page_source.h**
  外存数据源,用来创建DMatrix, 继承自*DataSource*.以sparse_batch_page二进制的存储. 
* **simple_dmatrix.h**
  DMatrix的内存(in-memory)版本.*SimpleDMatrix*继承自*DMatrix*.
* **sparse_page_dmatrix.h**
  DMatrix的外存(external-memory)版本.*SparsePageDMatrix*类继承自*DMatrix*
* **sparse_batch_page.h**
  可以被存储到磁盘的稀疏批处理内容持有者(content holder of sparse batch).这种表示方式
  在外存计算中可以被有效利用.定义了SparsePage类,这是一种稀疏批处理的内存中(in-memory)
  存储单元.
* **simple_dmatrix.cc**
  simple_dmatrix.h的定义.梯度提升的输入数据结构.
* **sparse_page_dmatrix.cc**
  外存版本的Page迭代器.
* **simple_csr_source.cc**
  simple_csr_source.h中的定义
* **sparse_page_source.cc**
  sparse_page_source.h中的定义
* **sparse_page_raw_format.cc**
  sparse page的原始二进制格式
* **sparse_page_writer.cc**
  sparse page的写者(Writer)类的定义.

### tree/
* **param.h**
  支持树构建的训练参数和统计量.包括:
  声明回归树的训练参数类(*TrainParam*);
  根据统计量计算损失函数的代价(cost of loss function)的函数CalcGain(),CalcWeights();
  用于构建树的核心统计类(包括梯度矩阵和Hessian矩阵)*GradStats*;
  有助于树划分方案存储和表示的统计类*SplitEntry*
  定义了用于vector容器的字符串序列化，目的是为了得到参数.命名空间位于std.
* **update_basemarker_inl.h**
  实现了一个通用的树构建器.包括:
  *BaseMaker*类.定义在构建树时必需的通用操作. 继承自*TreeUpdater*.
  大量的静态帮助函数,如初始化临时数据结构InitData(),
* **tree_model.cc**
  树的模型结构. 包括:
  转储回归树到文本的内部函数DumpRegTree
  转储模型函数DumpModel实现.内部调用DumpRegTree.
* **tree_updater.cc**
  树更新器的注册工厂.(Registry)
* **updater_colmaker.cc**
  使用逐列更新去构建树.包括:
  ColMarker类,继承自TreeUpdater,按行并行的生长树.
  DistColMarker类, 继承自ColMarker.
  * **updater_histmaker.cc**
    使用直方图计数(histogram counting)去构建树.包括:
    使用近似直方图构建树LocalHistMarker
    使用近似全局直方图建议(proposal of histogram)构建树.GlobalHistMaker
    使用近似全局直方图构建树HistMaker
 * **updater_prune.cc**
   根据给定的统计量进行剪枝
   定义了在当树生长完成之后进行剪枝的剪枝器TreePruner,继承自*TreeUpdater*
   尝试去剪掉当前叶子节点函数TryPruneLeat(),进行剪枝函数DoPrune()
 * **updater_refresh.cc**
   根据数据集对权重和统计量进行更新
   在剪枝后更新树的统计量和叶子节点的值TreeRefresher, 继承自TreeUpdater
 * **updater_skmaker.cc**
    使用近似素描(approximation sketch)构建一棵树,需要一个refresh使得统计量完全正确
 * **updater_sync.cc**
   在所有的分布式节点中同步树(synchronize the tree)
   定义了同步器TreeSyncher, 继承自*TreeUpdater*
   
  
### gbm/
* **gbm.cc**
  梯度提升器(gradient booster)的注册工厂(Registry).
* **gbtree.cc**
  梯度提升树模型的实现.
* **gblinear.cc**
  带L1/L2正则的梯度提升线性模型的实现,更新规则为坐标上升.
### metric/
* **metric.cc**
  评价矩阵的注册工厂(Registry).
* **multiclass-metric.cc**
  多分类的评价矩阵. 定义了多分类评价的基类
* **rank_metric.cc**
  基于预测排序的的矩阵，包括AUC for both classification and rank, Precision, 
  Normalized Discounted Cumulative Gain, Mean Average Precision(MAP) 
  for both classification and rank.
* **elementwise_metric.cc**
  逐元素二分类或回归的评价矩阵，包括RMSE, MAE, LogLoss(Negative log likelihood
  for logisitic regression), Negative Log Likelihood possion regression, Gamma
  Deviance, Binary classification error等. 
### objective/
* **objective.cc**
  所有目标函数的注册工厂(Registry).
* **multiclass_obj.cc**
  多分类目标函数的定义.包含输出类别标签的*Softmax*多分类函数和输出概率分布的
  *Softmax*多分类函数.
* **rank_obj.cc**
  排序损失函数的定义.包含逐对排序目标函数,LambdaRank with NDCG and with MAP
* **regression_obj.cc**
  单值回归和分类目标函数.包括线性回归, logistic regression 用来做概率回归任务, 
  logistic用来做二分类任务等.


















